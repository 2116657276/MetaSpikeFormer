#!/usr/bin/env python3
"""
MetaSpikeFormer — HASYv2 Math Symbol Classification (Mac MPS)
Optimized for Apple Silicon (M1 Pro 16GB, 6-8GB target).

HASYv2: 369 classes, ~168K grayscale 32x32 images.

Key MPS optimizations:
  - T=3 (reduced from 4 → 25% less activation memory)
  - batch_size=16 (verified best for SNN generalization)
  - num_workers=0 (MPS best practice)
  - Gradient checkpointing on SDSA blocks
  - GroupNorm for small-batch stability

Usage:
  # Dry-run: verify pipeline
  python mac/train_hasyv2.py --preset dryrun

  # Quick test: 10K samples, 3 epochs (~5 min)
  python mac/train_hasyv2.py --preset quick

  # Half data: 75K samples, 20 epochs (~6 hours)
  python mac/train_hasyv2.py --preset half

  # Full training: all 150K, 50 epochs (~35 hours)
  python mac/train_hasyv2.py --preset full
"""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import json

import torch
import torch.nn as nn

from model import (
    MetaSpikeFormer,
    meta_spikeformer_tiny,
    meta_spikeformer_hasyv2,
    meta_spikeformer_hasyv2_narrow,
)
from dataset_hasyv2 import build_hasyv2, HASYV2_NUM_CLASSES
from train import (
    train_one_epoch, evaluate, FiringRateMonitor, estimate_sops,
    estimate_memory, save_checkpoint, load_checkpoint,
    WarmupCosineScheduler, analyze_batch_records,
)
from spikingjelly.activation_based import functional, neuron


# ---------------------------------------------------------------------------
#  Memory-optimized model factory
# ---------------------------------------------------------------------------

def build_model(args):
    """Build and configure model for HASYv2 with memory optimizations."""
    model_map = {
        'tiny': meta_spikeformer_tiny,
        'narrow': meta_spikeformer_hasyv2_narrow,
        'hasyv2': meta_spikeformer_hasyv2,
    }
    builder = model_map[args.model_size]

    model_kwargs = dict(
        T=args.T,
        tau=args.tau,
        v_threshold=args.v_threshold,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        use_groupnorm=args.use_groupnorm,
    )

    model = builder(**model_kwargs)
    functional.set_step_mode(model, 'm')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = estimate_memory(model, args.batch_size, args.T, input_shape=(1, 32, 32))
    print(f"Model: {args.model_size}, {n_params/1e6:.2f}M params, T={args.T}")
    print(f"Memory: ~{mem['total_mb']:.0f}MB "
          f"(params:{mem['params_mb']:.0f} grads:{mem['grads_mb']:.0f} "
          f"optim:{mem['optim_mb']:.0f} act:{mem['peak_act_mb']:.0f})")

    return model


# ---------------------------------------------------------------------------
#  Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Meta-SpikeFormer HASYv2 (Mac MPS)')

    # Presets
    parser.add_argument('--preset', type=str, default='',
                        choices=['dryrun', 'quick', 'half', 'full', ''],
                        help='dryrun=verify pipeline, quick=10K/3ep, half=75K/20ep, full=150K/50ep')

    # Model
    parser.add_argument('--model_size', type=str, default='narrow',
                        choices=['tiny', 'narrow', 'hasyv2'])
    parser.add_argument('--T', type=int, default=3,
                        help='SNN time steps (3 saves 25%% mem vs 4)')
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v_threshold', type=float, default=0.3)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)
    parser.add_argument('--use_groupnorm', action='store_true', default=True)
    parser.add_argument('--no_groupnorm', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='0=use all available (150K)')
    parser.add_argument('--max_val_samples', type=int, default=0)
    parser.add_argument('--val_split', type=float, default=0.10,
                        help='[deprecated] HASYv2 uses fold-1 split')

    # System
    parser.add_argument('--num_workers', type=int, default=0,
                        help='MPS: keep at 0')
    parser.add_argument('--device', type=str, default='mps')

    # Logging
    parser.add_argument('--log_csv', type=str, default='./logs/hasyv2_mac.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='Stop if no val improvement for N epochs (0=disable)')

    # Data
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    # ---- Apply presets ----
    if args.preset == 'dryrun':
        args.model_size = 'tiny'
        args.epochs = 1
        args.batch_size = 4
        args.max_train_samples = 100
        args.max_val_samples = 50
        args.warmup_epochs = 0
        args.T = 3
        print("🚀 dryrun: tiny, 1 batch verify")
    elif args.preset == 'quick':
        args.model_size = 'tiny'
        args.epochs = 5
        args.batch_size = 16
        args.max_train_samples = 10000
        args.max_val_samples = 2000
        args.warmup_epochs = 1
        args.T = 3
        print("🚀 quick: tiny, 5 epochs, 10K samples")
    elif args.preset == 'half':
        args.model_size = 'narrow'
        args.epochs = 30
        args.batch_size = 16
        args.max_train_samples = 75000
        args.max_val_samples = 15000
        args.warmup_epochs = 3
        args.T = 3
        args.early_stop = 8
        print("🚀 half: narrow (6.7M), 30 epochs, 75K samples, early_stop=8")
    elif args.preset == 'full':
        args.model_size = 'hasyv2'
        args.epochs = 40
        args.batch_size = 16
        args.max_train_samples = 0
        args.max_val_samples = 0
        args.warmup_epochs = 5
        args.T = 3
        args.early_stop = 10
        print("🚀 full: hasyv2 (13.3M), 40 epochs, 151K samples, early_stop=10")

    return args


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    use_gn = not args.no_groupnorm
    args.use_groupnorm = use_gn

    # Device
    device = torch.device(args.device)
    print(f"Device: {device} (Apple Silicon)")
    print(f"Config: T={args.T}, batch={args.batch_size}, "
          f"v_th={args.v_threshold}, lr={args.lr}")

    # ---- Data ----
    print("Loading HASYv2...")
    train_loader, val_loader, num_classes = build_hasyv2(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        T=args.T,
        root=args.data_root,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print(f"Classes: {num_classes}, Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    # ---- Model ----
    print("Building model...")
    model = build_model(args).to(device)

    # ---- Monitor ----
    fr_monitor = FiringRateMonitor(model)
    print(f"FR monitor: {len(fr_monitor.hooks)} LIF nodes")

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs,
        base_lr=args.lr, min_lr=args.min_lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Save dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV Logger ----
    log_dir = Path(args.log_csv).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(args.log_csv, 'w') if args.log_csv else None
    batch_records = []
    if csv_file:
        csv_file.write('epoch,batch,loss,acc,lr,grad_norm,fr\n')

    def batch_logger(epoch, batch_idx, loss, acc, lr, grad_norm, fr):
        if csv_file:
            csv_file.write(
                f'{epoch},{batch_idx},{loss:.6f},{acc:.4f},'
                f'{lr:.8f},{grad_norm:.4f},{fr:.6f}\n')
        batch_records.append({
            'epoch': epoch, 'batch': batch_idx, 'loss': loss,
            'acc': acc, 'lr': lr, 'grad_norm': grad_norm, 'fr': fr})

    # ---- Resume ----
    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ---- Training ----
    print(f"\n=== Training ({args.epochs} epochs, {len(train_loader)} batches/epoch) ===")
    t_total_start = time.time()

    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            grad_clip=args.grad_clip, fr_monitor=fr_monitor, dry_run=False,
            batch_logger=batch_logger if csv_file else None)

        # Eval
        val_metrics = evaluate(model, val_loader, criterion, device,
                               fr_monitor=fr_monitor)

        epoch_time = time.time() - t0

        # FR + SOPs
        avg_fr = fr_monitor.get_avg_firing_rate()
        sops = estimate_sops(model, avg_fr, T=args.T)
        health = fr_monitor.check_health()
        fr_monitor.clear()

        # Checkpoint
        is_best = val_metrics['acc'] > best_acc
        if is_best:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          val_metrics,
                          str(save_dir / f'hasyv2_{args.model_size}_best.pt'))
            print(f"  → Best saved ({best_acc:.2f}%)")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          val_metrics,
                          str(save_dir / f'hasyv2_{args.model_size}_ep{epoch+1}.pt'))

        # Log
        health_str = f" [{health}]" if health else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | LR: {current_lr:.6f} | "
              f"Train: {train_metrics['loss']:.4f} {train_metrics['acc']:.2f}% | "
              f"Val: {val_metrics['loss']:.4f} {val_metrics['acc']:.2f}% | "
              f"FR: {avg_fr:.4f}{health_str} | SOPs: {sops:.2f}M | "
              f"Time: {epoch_time:.0f}s {'*' if is_best else ''}")

        # Early stopping
        if is_best:
            no_improve_count = 0
        else:
            no_improve_count += 1
        if args.early_stop > 0 and no_improve_count >= args.early_stop:
            print(f"⚠ Early stop: no improvement for {no_improve_count} epochs")
            break

    # ---- Done ----
    total_time = time.time() - t_total_start
    print(f"\n=== Done ({total_time/3600:.1f}h) ===")
    print(f"Best Val Acc: {best_acc:.2f}%")

    # Save final
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_acc, {},
                  str(save_dir / f'hasyv2_{args.model_size}_final.pt'))

    # Analysis
    if csv_file:
        csv_file.close()
        analysis = analyze_batch_records(batch_records)
        analysis_path = args.log_csv.replace('.csv', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Logs: {args.log_csv}, {analysis_path}")
        print(f"\n=== Analysis ===")
        print(f"Total batches: {analysis['total_batches']}")
        print(f"Loss: {analysis['loss_start']:.4f} → {analysis['loss_end']:.4f} "
              f"(Δ: {analysis['loss_delta']:+.4f})")
        print(f"Acc: {analysis['acc_start']:.1f}% → {analysis['acc_end']:.1f}% "
              f"(Δ: {analysis['acc_delta']:+.1f}%)")
        print(f"FR: {analysis['fr_mean']:.4f} ± {analysis['fr_std']:.4f}")


if __name__ == '__main__':
    main()
