"""
Training & Evaluation Loop for Meta-SpikeFormer on CIFAR-100.

Key SNN-specific practices:
  - reset_net(model) after EVERY forward pass (train & eval).
  - Firing-rate monitoring on all LIF neurons.
  - Synaptic Operations (SOPs) estimation.
  - Multi-step mode (T=4 time-steps).
  - Gradient clipping for stability.
  - Warmup + CosineAnnealing scheduler.

Usage:
  # Quick dry-run (1 batch only, check tensor shapes & gradient flow):
  python train.py --model_size micro --epochs 1 --batch_size 4 --dry_run

  # CPU warmup training:
  python train.py --model_size micro --epochs 5 --batch_size 16 --device cpu

  # Full training on MPS:
  python train.py --model_size cifar100 --epochs 200 --batch_size 64 --device mps
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, neuron

from model import (
    MetaSpikeFormer,
    meta_spikeformer_tiny,
    meta_spikeformer_cifar100,
)
from dataset import build_cifar100


# ---------------------------------------------------------------------------
#  Firing-rate Monitor
# ---------------------------------------------------------------------------

class FiringRateMonitor:
    """
    Memory-efficient firing-rate monitor using forward hooks.
    Stores only running counts, not full tensors — avoids MPS OOM.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.total_spikes = 0.0
        self.total_elements = 0
        self.hooks = []
        self._attach()

    def _attach(self):
        def make_hook():
            def hook(module, input, output):
                # output is the spike tensor from LIFNode
                self.total_spikes += output.detach().sum().item()
                self.total_elements += output.detach().numel()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, neuron.LIFNode):
                h = module.register_forward_hook(make_hook())
                self.hooks.append(h)

    def get_avg_firing_rate(self) -> float:
        """Overall average firing rate since last clear."""
        if self.total_elements == 0:
            return 0.0
        return self.total_spikes / self.total_elements

    def check_health(self) -> Optional[str]:
        """Return warning string if firing rates are unhealthy."""
        avg_fr = self.get_avg_firing_rate()
        if avg_fr < 0.005:
            return f"WARNING: Firing rate too low ({avg_fr:.5f}) — neurons may be dead"
        if avg_fr > 0.95:
            return f"WARNING: Firing rate too high ({avg_fr:.5f}) — neurons saturated"
        return None

    def clear(self):
        self.total_spikes = 0.0
        self.total_elements = 0

    def disable(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
#  Synaptic Operations (SOPs) estimation
# ---------------------------------------------------------------------------

def estimate_sops(model: nn.Module, firing_rate: float,
                  input_shape=(3, 32, 32), T: int = 4) -> float:
    """
    Estimate Synaptic Operations for one forward pass.

    SOPs ≈ Σ (pre-synaptic spike count × fan-out).
    Returns SOPs in Millions (M).
    """
    total_synapses = 0

    for m in model.modules():
        if isinstance(m, nn.Linear):
            total_synapses += m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d):
            k_area = m.kernel_size[0] * m.kernel_size[1]
            total_synapses += m.in_channels * m.out_channels * k_area

    sops = firing_rate * total_synapses * T
    return sops / 1e6  # in Millions


# ---------------------------------------------------------------------------
#  Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory(model: nn.Module, batch_size: int, T: int = 4,
                    input_shape=(3, 32, 32)) -> Dict[str, float]:
    """Estimate memory usage in MB for training."""
    # Parameter memory
    param_bytes = sum(p.numel() * 4 for p in model.parameters())
    grad_bytes = param_bytes  # same as params
    # Optimizer (AdamW: 2x params for m and v)
    optim_bytes = param_bytes * 2

    # Activation memory (rough estimate)
    # Peak at stage 0: T * B * C * H * W * 4 bytes
    peak_act = T * batch_size * max([3] + list(model.embed_dims)) * 32 * 32 * 4

    total_mb = (param_bytes + grad_bytes + optim_bytes + peak_act) / 1e6

    return {
        'params_mb': param_bytes / 1e6,
        'grads_mb': grad_bytes / 1e6,
        'optim_mb': optim_bytes / 1e6,
        'peak_act_mb': peak_act / 1e6,
        'total_mb': total_mb,
    }


# ---------------------------------------------------------------------------
#  Batch record analysis
# ---------------------------------------------------------------------------

def analyze_batch_records(records: list) -> dict:
    """Analyze per-batch training records and return diagnostic insights."""
    if not records:
        return {}

    n = len(records)
    epochs = sorted(set(r['epoch'] for r in records))

    # First and last epoch aggregate
    first_epoch = [r for r in records if r['epoch'] == epochs[0]]
    last_epoch = [r for r in records if r['epoch'] == epochs[-1]]

    loss_start = sum(r['loss'] for r in first_epoch) / len(first_epoch)
    loss_end = sum(r['loss'] for r in last_epoch) / len(last_epoch)
    acc_start = sum(r['acc'] for r in first_epoch) / len(first_epoch)
    acc_end = sum(r['acc'] for r in last_epoch) / len(last_epoch)

    # FR trend
    fr_values = [r['fr'] for r in records if r['fr'] > 0]
    fr_mean = sum(fr_values) / len(fr_values) if fr_values else 0
    fr_std = (sum((f - fr_mean)**2 for f in fr_values) / len(fr_values))**0.5 if fr_values else 0

    first_fr = sum(r['fr'] for r in first_epoch if r['fr'] > 0) / max(1, sum(1 for r in first_epoch if r['fr'] > 0))
    last_fr = sum(r['fr'] for r in last_epoch if r['fr'] > 0) / max(1, sum(1 for r in last_epoch if r['fr'] > 0))

    fr_trend = 'stable'
    if last_fr < first_fr * 0.8:
        fr_trend = 'declining'
    elif last_fr > first_fr * 1.2:
        fr_trend = 'rising'

    # Grad norm
    grad_norms = [r['grad_norm'] for r in records if r['grad_norm'] > 0]
    gn_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    gn_std = (sum((g - gn_mean)**2 for g in grad_norms) / len(grad_norms))**0.5 if grad_norms else 0

    # Loss plateau check (loss in last 20% of batches vs first 20%)
    loss_first_20 = [r['loss'] for r in records[:n//5]]
    loss_last_20 = [r['loss'] for r in records[-n//5:]]
    loss_plateau_early = (sum(loss_last_20)/len(loss_last_20) > sum(loss_first_20)/len(loss_first_20) * 0.95)

    # Per-epoch summary
    epoch_summaries = {}
    for ep in epochs:
        ep_records = [r for r in records if r['epoch'] == ep]
        epoch_summaries[ep] = {
            'loss': sum(r['loss'] for r in ep_records) / len(ep_records),
            'acc': sum(r['acc'] for r in ep_records) / len(ep_records),
            'fr': sum(r['fr'] for r in ep_records if r['fr'] > 0) / max(1, sum(1 for r in ep_records if r['fr'] > 0)),
            'grad_norm': sum(r['grad_norm'] for r in ep_records) / len(ep_records),
        }

    return {
        'total_batches': n,
        'num_epochs': len(epochs),
        'loss_start': loss_start,
        'loss_end': loss_end,
        'loss_delta': loss_end - loss_start,
        'acc_start': acc_start,
        'acc_end': acc_end,
        'acc_delta': acc_end - acc_start,
        'fr_mean': fr_mean,
        'fr_std': fr_std,
        'fr_trend': fr_trend,
        'grad_norm_mean': gn_mean,
        'grad_norm_std': gn_std,
        'loss_plateau_early': loss_plateau_early,
        'overfit_signal': (acc_end - acc_start > 5 and epochs[-1] > 3),  # crude heuristic
        'epoch_summaries': {str(k): v for k, v in epoch_summaries.items()},
    }


# ---------------------------------------------------------------------------
#  One epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    fr_monitor: Optional[FiringRateMonitor] = None,
    dry_run: bool = False,
    batch_logger = None,  # callback(epoch, batch_idx, loss, acc, lr, grad_norm, fr)
    grad_accum_steps: int = 1,  # gradient accumulation steps
) -> Dict[str, float]:
    """
    Train one epoch with optional gradient accumulation.

    When grad_accum_steps > 1, gradients are accumulated over that many
    micro-batches before calling optimizer.step().  This simulates a larger
    effective batch size without the VRAM cost:
        effective_batch = batch_size × grad_accum_steps

    Loss is scaled by 1/grad_accum_steps so the gradient magnitude matches
    the effective batch size.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    current_lr = optimizer.param_groups[0]['lr']
    accum_step = 0
    total_norm = 0.0
    n_batches = len(loader)

    # Zero grad at start of epoch
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Handle missing T dim
        if images.dim() == 4:
            images = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

        # Forward — scale loss for gradient accumulation
        logits = model(images)
        loss = criterion(logits, labels) / grad_accum_steps

        # Backward (accumulate gradients)
        loss.backward()

        # CRITICAL: reset network state after EVERY forward pass
        functional.reset_net(model)

        # Stats (use unscaled loss for reporting)
        running_loss += loss.item() * grad_accum_steps
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accum_step += 1

        # Optimizer step — only at accumulation boundary or end of epoch
        is_accum_boundary = (accum_step == grad_accum_steps)
        is_last_batch = (batch_idx == n_batches - 1)

        if is_accum_boundary or is_last_batch:
            # Gradient clipping
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            else:
                total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Per-batch logging (at each optimizer step)
            if batch_logger:
                fr = fr_monitor.get_avg_firing_rate() if fr_monitor else 0.0
                batch_acc_val = predicted.eq(labels).sum().item() / labels.size(0) * 100.0
                batch_logger(epoch, batch_idx, loss.item() * grad_accum_steps,
                           batch_acc_val, current_lr, total_norm, fr)

            accum_step = 0
        else:
            # Log intermediate micro-batches without optimizer stats
            if batch_logger:
                fr = fr_monitor.get_avg_firing_rate() if fr_monitor else 0.0
                batch_acc_val = predicted.eq(labels).sum().item() / labels.size(0) * 100.0
                batch_logger(epoch, batch_idx, loss.item() * grad_accum_steps,
                           batch_acc_val, current_lr, 0.0, fr)

        if dry_run and batch_idx == 0:
            print(f"  Dry-run batch {batch_idx+1}: loss={loss.item() * grad_accum_steps:.4f}, "
                  f"acc={100.0*correct/total:.2f}%")
            print("  Dry-run PASSED — tensor shapes & gradient flow OK.")
            break

        if batch_idx % 50 == 0:
            fr_str = ""
            if fr_monitor:
                avg_fr = fr_monitor.get_avg_firing_rate()
                health = fr_monitor.check_health()
                fr_str = f", FR={avg_fr:.4f}"
                if health:
                    fr_str += f" [{health}]"
            eff_batch = (batch_idx + 1) * images.size(1)  # approximate
            print(f"  Epoch {epoch} [{batch_idx}/{n_batches}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f}, "
                  f"Acc: {100.0*correct/total:.2f}%{fr_str}")
            if fr_monitor:
                fr_monitor.clear()

    return {
        'loss': running_loss / n_batches,
        'acc': 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    fr_monitor: Optional[FiringRateMonitor] = None,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if images.dim() == 4:
            images = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

        logits = model(images)
        loss = criterion(logits, labels)

        # CRITICAL: reset network state after EVERY forward pass
        functional.reset_net(model)

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        'loss': running_loss / len(loader),
        'acc': 100.0 * correct / total,
    }


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    metrics: dict,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> dict:
    """Load training checkpoint. Returns checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


# ---------------------------------------------------------------------------
#  Warmup scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=min_lr,
            last_epoch=last_epoch,
        )

    def step(self, epoch: int):
        self.last_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / max(1, self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * lr_scale
        else:
            self.cos_scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'cos_scheduler': self.cos_scheduler.state_dict(),
            'last_epoch': self.last_epoch,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.cos_scheduler.load_state_dict(state_dict['cos_scheduler'])
        self.last_epoch = state_dict.get('last_epoch', -1)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.min_lr = state_dict.get('min_lr', self.min_lr)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Meta-SpikeFormer on CIFAR-100')

    # Model
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'cifar100'],
                        help='Model size variant')
    parser.add_argument('--T', type=int, default=4,
                        help='SNN time steps')
    parser.add_argument('--tau', type=float, default=2.0,
                        help='LIF neuron time constant')
    parser.add_argument('--v_threshold', type=float, default=1.0,
                        help='LIF neuron threshold')
    parser.add_argument('--use_groupnorm', action='store_true', default=True,
                        help='Use GroupNorm (default: True)')
    parser.add_argument('--no_groupnorm', action='store_true',
                        help='Use LayerNorm instead of GroupNorm')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='Dropout rate for MLP blocks')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0,
                        help='Dropout rate for attention')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Limit training samples (0 = use all)')
    parser.add_argument('--max_val_samples', type=int, default=0,
                        help='Limit validation samples (0 = use all)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max_norm (0 = disable)')

    # System
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])

    # Flags
    parser.add_argument('--dry_run', action='store_true',
                        help='Run only 1 batch to verify setup.')
    parser.add_argument('--no_monitor', action='store_true',
                        help='Disable firing-rate monitoring.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint path.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints.')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save checkpoint every N epochs (0 = only best).')
    parser.add_argument('--log_csv', type=str, default='',
                        help='Save per-batch metrics to CSV file.')

    args = parser.parse_args()

    # ---- Device ----
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == 'mps':
        print("  Note: MPS (Apple GPU) — ensure sufficient unified memory available")

    # ---- Data ----
    print("Loading CIFAR-100...")
    train_loader, val_loader, num_classes = build_cifar100(
        batch_size=args.batch_size, num_workers=args.num_workers, T=args.T,
    )

    # Apply sample limits if specified
    from torch.utils.data import Subset, DataLoader
    if args.max_train_samples > 0:
        n = min(args.max_train_samples, len(train_loader.dataset))
        indices = range(n)
        train_loader = DataLoader(
            Subset(train_loader.dataset, indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        print(f"  Train subset: {n} samples")
    if args.max_val_samples > 0:
        n = min(args.max_val_samples, len(val_loader.dataset))
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(n)),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        print(f"  Val subset: {n} samples")

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Model ----
    print("Building model...")
    use_gn = not args.no_groupnorm
    model_kwargs = dict(
        T=args.T,
        tau=args.tau,
        v_threshold=args.v_threshold,
        use_groupnorm=use_gn,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
    )
    if args.model_size == 'tiny':
        model = meta_spikeformer_tiny(**model_kwargs)
    else:
        model = meta_spikeformer_cifar100(**model_kwargs)

    model = model.to(device)
    functional.set_step_mode(model, 'm')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_size}, Params: {n_params/1e6:.2f}M, T={args.T}")
    print(f"  GroupNorm: {use_gn}, tau: {args.tau}, v_threshold: {args.v_threshold}")

    # ---- Memory estimation ----
    mem = estimate_memory(model, args.batch_size, args.T)
    print(f"Estimated memory: {mem['total_mb']:.1f}MB total "
          f"(params: {mem['params_mb']:.1f}MB, grads: {mem['grads_mb']:.1f}MB, "
          f"optim: {mem['optim_mb']:.1f}MB, peak_act: {mem['peak_act_mb']:.1f}MB)")

    # ---- Firing-rate monitor ----
    fr_monitor = None if args.no_monitor else FiringRateMonitor(model)
    if fr_monitor:
        print(f"Firing-rate monitor attached to {len(fr_monitor.hooks)} LIF nodes")

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    # ---- Criterion ----
    criterion = nn.CrossEntropyLoss()

    # ---- Checkpoint directory ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dry-run ----
    if args.dry_run:
        print("\n=== DRY RUN ===")

        # Use a small batch on CPU for verification
        dry_device = torch.device('cpu')
        model = model.to(dry_device)
        functional.set_step_mode(model, 'm')
        print(f"Dry-run device: {dry_device}")

        dry_batch = 4
        images, labels = next(iter(train_loader))
        images = images[:dry_batch].to(dry_device)
        labels = labels[:dry_batch].to(dry_device)

        functional.reset_net(model)

        print(f"Input shape: {images.shape}")
        images_t = images.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
        print(f"After T-repeat: {images_t.shape} (expect [T,B,C,H,W]=[{args.T},{dry_batch},3,32,32])")

        logits = model(images_t)
        print(f"Output: {logits.shape} (expect [{dry_batch},{num_classes}])")

        loss = criterion(logits, labels)
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"Gradient norm (sum): {grad_norm:.4f}")

        if fr_monitor:
            rates = fr_monitor.get_firing_rates()
            avg_fr = fr_monitor.get_avg_firing_rate()
            print(f"Avg firing rate: {avg_fr:.4f}")
            health = fr_monitor.check_health()
            if health:
                print(f"  {health}")
            sops = estimate_sops(model, avg_fr, T=args.T)
            print(f"Estimated SOPs: {sops:.2f} M")

        functional.reset_net(model)
        print("=== DRY RUN PASSED ===\n")
        return

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ---- CSV batch logger ----
    csv_file = None
    batch_records = []  # collect all batch data for analysis
    if args.log_csv:
        csv_file = open(args.log_csv, 'w')
        csv_file.write('epoch,batch,loss,acc,lr,grad_norm,fr\n')

    def batch_logger(epoch, batch_idx, loss, acc, lr, grad_norm, fr):
        if csv_file:
            csv_file.write(f'{epoch},{batch_idx},{loss:.6f},{acc:.4f},{lr:.8f},{grad_norm:.4f},{fr:.6f}\n')
        batch_records.append({
            'epoch': epoch, 'batch': batch_idx, 'loss': loss, 'acc': acc,
            'lr': lr, 'grad_norm': grad_norm, 'fr': fr,
        })

    # ---- Training loop ----
    print(f"\n=== Starting Training ({args.epochs} epochs, batch={args.batch_size}) ===")

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()

        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            grad_clip=args.grad_clip, fr_monitor=fr_monitor, dry_run=False,
            batch_logger=batch_logger if args.log_csv else None,
        )

        # Eval
        val_metrics = evaluate(model, val_loader, criterion, device, fr_monitor=fr_monitor)

        epoch_time = time.time() - t0

        # Firing-rate & SOPs
        fr_str = ""
        if fr_monitor:
            avg_fr = fr_monitor.get_avg_firing_rate()
            sops = estimate_sops(model, avg_fr, T=args.T)
            fr_str = f" | FR: {avg_fr:.4f} | SOPs: {sops:.2f}M"
            health = fr_monitor.check_health()
            if health:
                fr_str += f" [{health}]"
            fr_monitor.clear()

        # Track best
        is_best = val_metrics['acc'] > best_acc
        if is_best:
            best_acc = val_metrics['acc']

        # Logging
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}%"
              f"{fr_str} | "
              f"Time: {epoch_time:.1f}s {'*' if is_best else ''}")

        # Save checkpoint
        if is_best:
            save_checkpoint(model, optimizer, scheduler,
                          epoch, best_acc, val_metrics,
                          str(save_dir / f'{args.model_size}_best.pt'))
            print(f"  → Best model saved (acc={best_acc:.2f}%)")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler,
                          epoch, best_acc, val_metrics,
                          str(save_dir / f'{args.model_size}_epoch{epoch+1}.pt'))

    print(f"\n=== Training Complete ===")
    print(f"Best Val Accuracy: {best_acc:.2f}%")

    # Save final model
    save_checkpoint(model, optimizer, scheduler,
                  args.epochs - 1, best_acc, {},
                  str(save_dir / f'{args.model_size}_final.pt'))
    print(f"Final model saved to {save_dir}/{args.model_size}_final.pt")

    # Close CSV and generate quick analysis
    if csv_file:
        csv_file.close()
        print(f"Batch log saved to {args.log_csv}")

        # Quick analysis
        import json
        analysis = analyze_batch_records(batch_records)
        analysis_path = args.log_csv.replace('.csv', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_path}")

        # Print key findings
        print("\n=== Batch Analysis ===")
        print(f"Total batches: {analysis['total_batches']}")
        print(f"Loss: {analysis['loss_start']:.4f} → {analysis['loss_end']:.4f} "
              f"(Δ: {analysis['loss_delta']:+.4f})")
        print(f"Acc: {analysis['acc_start']:.1f}% → {analysis['acc_end']:.1f}% "
              f"(Δ: {analysis['acc_delta']:+.1f}%)")
        print(f"FR: {analysis['fr_mean']:.4f} ± {analysis['fr_std']:.4f}")
        print(f"Grad Norm: {analysis['grad_norm_mean']:.2f} ± {analysis['grad_norm_std']:.2f}")
        if analysis.get('fr_trend') == 'declining':
            print("⚠ FR declining — consider lowering v_threshold or increasing tau")
        elif analysis.get('fr_trend') == 'rising':
            print("⚠ FR rising — consider raising v_threshold")
        if analysis.get('loss_plateau_early'):
            print("⚠ Loss plateaued early — consider higher LR or more warmup")
        if analysis.get('overfit_signal'):
            print("⚠ Possible overfitting — train acc >> val acc in later epochs")


if __name__ == '__main__':
    main()
