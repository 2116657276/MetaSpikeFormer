#!/usr/bin/env python3
"""
MetaSpikeFormer — HASYv2 Math Symbol Classification (CUDA RTX 4060)

HASYv2: 369 classes, ~168K grayscale 32x32 images.
VRAM limit: <6GB on RTX 4060 (8GB).

Usage:
  python cuda/train_hasyv2.py --quick              # shallow, 10K, 10 ep (~5min)
  python cuda/train_hasyv2.py --half               # shallow, 75K, 25 ep (~9h)
"""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure progress prints are visible when stdout is redirected
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

import argparse
import time
import json

import torch
import torch.nn as nn

from model import (
    meta_spikeformer_hasyv2,
    meta_spikeformer_hasyv2_narrow,
    meta_spikeformer_hasyv2_shallow,
)
from dataset_hasyv2 import build_hasyv2
from train import (
    train_one_epoch, evaluate, FiringRateMonitor, estimate_sops,
    estimate_memory, save_checkpoint, load_checkpoint,
    WarmupCosineScheduler, analyze_batch_records,
)
from spikingjelly.activation_based import functional


# ---------------------------------------------------------------------------
#  Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='MetaSpikeFormer HASYv2 (CUDA RTX 4060)')

    # ---- Presets ----
    parser.add_argument('--quick', action='store_true',
                        help='Quick: shallow PLIF, 10K/10ep, ~5 min')
    parser.add_argument('--half', action='store_true',
                        help='Half: shallow PLIF, 75K/25ep, ~9 h')
    parser.add_argument('--narrow', action='store_true',
                        help='Narrow PLIF: narrow 6.7M, 75K/25ep, T=3')

    # ---- Model ----
    parser.add_argument('--model_size', type=str, default='shallow',
                        choices=['shallow', 'narrow', 'hasyv2'])
    parser.add_argument('--T', type=int, default=3, help='SNN time steps')
    parser.add_argument('--tau', type=float, default=2.0, help='LIF time constant')
    parser.add_argument('--v_threshold', type=float, default=0.20,
                        help='LIF threshold (0.20: between Mac 0.25 & CUDA 0.15)')
    parser.add_argument('--drop_rate', type=float, default=0.05)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # ---- Training ----
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='0=use all available')
    parser.add_argument('--max_val_samples', type=int, default=0)

    # ---- V-based regularization (lightweight — only activates when V drops) ----
    parser.add_argument('--lambda_v', type=float, default=0.3,
                        help='V-based reg weight (0=disabled)')
    parser.add_argument('--lambda_vneg', type=float, default=0.1,
                        help='Negative V penalty weight')

    # ---- Early stopping ----
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop if no val improvement for N epochs (0=disable)')
    parser.add_argument('--fr_critical', type=float, default=0.03,
                        help='Auto-stop if FR drops below this')

    # ---- System ----
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # ---- Logging ----
    parser.add_argument('--log_csv', type=str, default='./logs/hasyv2_shallow.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    # Apply presets
    if args.quick:
        args.model_size = 'shallow'
        args.epochs = 10
        args.max_train_samples = 10000
        args.max_val_samples = 2000
        args.save_every = 0
        args.early_stop_patience = 0
        args.log_csv = './logs/hasyv2_shallow_quick.csv'
        print("🧪 Quick: shallow PLIF, 10K train, 10 epochs")

    if args.half:
        args.model_size = 'shallow'
        args.epochs = 25
        args.max_train_samples = 75000
        args.max_val_samples = 0
        args.save_every = 5
        args.early_stop_patience = 10
        args.log_csv = './logs/hasyv2_shallow_half.csv'
        print("🚀 Half: shallow PLIF, 75K train, 25 epochs")

    if args.narrow:
        args.model_size = 'narrow'
        args.T = 3                     # T=3: spikes propagate through all 10 blocks
        args.epochs = 25
        args.max_train_samples = 75000
        args.max_val_samples = 0
        args.save_every = 5
        args.early_stop_patience = 10
        args.log_csv = './logs/hasyv2_narrow_plif.csv'
        print("🚀 Narrow: narrow PLIF 6.7M, 75K train, 25 epochs, T=3")

    return args


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ---- Device ----
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    print(f"Device: {device} — {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_total:.1f} GB | limit: 6GB")

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
    print(f"{num_classes} classes | Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ---- Model ----
    print("Building model...")
    model_map = {
        'shallow': meta_spikeformer_hasyv2_shallow,
        'narrow': meta_spikeformer_hasyv2_narrow,
        'hasyv2': meta_spikeformer_hasyv2,
    }
    model = model_map[args.model_size](
        T=args.T, tau=args.tau, v_threshold=args.v_threshold,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        use_groupnorm=True, use_plif=True,
    ).to(device)
    functional.set_step_mode(model, 'm')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = estimate_memory(model, args.batch_size, args.T, input_shape=(1, 32, 32))
    print(f"Model: {args.model_size}, {n_params/1e6:.2f}M params, T={args.T}")
    print(f"Est. memory: ~{mem['total_mb']:.0f}MB (VRAM<6GB: OK)")

    # ---- VRAM pre-check ----
    vram_mb = torch.cuda.memory_allocated() / 1e6
    vram_limit_mb = 6000
    if vram_mb > vram_limit_mb:
        raise RuntimeError(f"VRAM {vram_mb:.0f}MB exceeds {vram_limit_mb}MB limit!")
    print(f"VRAM after model load: {vram_mb:.0f}MB")

    # ---- Monitor ----
    fr_monitor = FiringRateMonitor(model)
    print(f"FR monitor: {len(fr_monitor.hooks)} nodes")

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
    log_dir = Path(args.log_csv).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV ----
    csv_file = open(args.log_csv, 'w') if args.log_csv else None
    batch_records = []
    if csv_file:
        csv_file.write('epoch,batch,loss,acc,lr,grad_norm,fr,ce_loss,v_low,v_neg\n')

    def batch_logger(epoch, batch_idx, loss, acc, lr, grad_norm, fr,
                     ce_loss=0.0, sd_loss=0.0, mp_loss=0.0):
        if csv_file:
            csv_file.write(
                f'{epoch},{batch_idx},{loss:.6f},{acc:.4f},'
                f'{lr:.8f},{grad_norm:.4f},{fr:.6f},'
                f'{ce_loss:.6f},{sd_loss:.6f},{mp_loss:.6f}\n')
        batch_records.append({
            'epoch': epoch, 'batch': batch_idx, 'loss': loss,
            'acc': acc, 'lr': lr, 'grad_norm': grad_norm, 'fr': fr,
            'ce_loss': ce_loss, 'v_low_loss': sd_loss, 'v_neg_loss': mp_loss})

    # ---- Resume ----
    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ---- Training ----
    print(f"\n=== Training ({args.epochs} epochs, {len(train_loader)} batches/epoch, "
          f"early_stop={args.early_stop_patience}) ===")

    no_improve = 0
    best_model_path = str(save_dir / f'hasyv2_{args.model_size}_best.pt')
    t_total = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            grad_clip=args.grad_clip, fr_monitor=fr_monitor, dry_run=False,
            batch_logger=batch_logger if csv_file else None,
            grad_accum_steps=1,
            lambda_sd=args.lambda_v, lambda_mp=args.lambda_vneg,
            target_fr_min=0.10)

        val_metrics = evaluate(model, val_loader, criterion, device,
                               fr_monitor=fr_monitor)
        epoch_time = time.time() - t0

        avg_fr = fr_monitor.get_avg_firing_rate()
        sops = estimate_sops(model, avg_fr, T=args.T, input_shape=(1, 32, 32))
        fr_monitor.clear()

        vram_mb = torch.cuda.memory_allocated() / 1e6

        # Checkpoint
        is_best = val_metrics['acc'] > best_acc + 0.001
        if is_best:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          val_metrics, best_model_path)
            print(f"  → Best saved ({best_acc:.2f}%)")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          val_metrics,
                          str(save_dir / f'hasyv2_{args.model_size}_ep{epoch+1}.pt'))

        # Log
        fr_flag = ""
        if avg_fr < args.fr_critical:
            fr_flag = " 💀FR CRITICAL!"
        elif avg_fr < 0.05:
            fr_flag = " ⚠️FR LOW"

        print(f"Epoch {epoch+1:3d}/{args.epochs} | LR: {current_lr:.6f} | "
              f"Train: {train_metrics['loss']:.4f} {train_metrics['acc']:.2f}% | "
              f"Val: {val_metrics['loss']:.4f} {val_metrics['acc']:.2f}% | "
              f"FR: {avg_fr:.4f}{fr_flag} | SOPs: {sops:.2f}M | "
              f"VRAM: {vram_mb:.0f}MB | {epoch_time:.0f}s {'*' if is_best else ''}")

        # Early stopping
        no_improve = 0 if is_best else no_improve + 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"⏹ Early stop ({no_improve} epochs no improvement)")
            break

        if avg_fr < args.fr_critical and epoch > args.warmup_epochs:
            print(f"💀 FR {avg_fr:.4f} < {args.fr_critical} — restoring best")
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            break

        if vram_mb > vram_limit_mb:
            print(f"🛑 VRAM {vram_mb:.0f}MB > {vram_limit_mb}MB — stopping")
            break

    # ---- Done ----
    print(f"\n=== Done ({ (time.time()-t_total)/3600:.1f}h) | Best Val: {best_acc:.2f}% ===")

    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_acc, {},
                  str(save_dir / f'hasyv2_{args.model_size}_final.pt'))

    if csv_file:
        csv_file.close()
        if batch_records:
            analysis = analyze_batch_records(batch_records)
            analysis_path = args.log_csv.replace('.csv', '_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Logs: {args.log_csv}, {analysis_path}")


if __name__ == '__main__':
    main()
