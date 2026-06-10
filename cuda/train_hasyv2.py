#!/usr/bin/env python3
"""
MetaSpikeFormer — HASYv2 Math Symbol Classification (CUDA RTX 4060)
Optimized for NVIDIA RTX 4060 (8GB VRAM) on WSL/Ubuntu.

HASYv2: 369 classes, ~168K grayscale 32x32 images.

Key features (v4):
  - Spike density + membrane potential regularization (prevents FR collapse)
  - Three model sizes: tiny (1.0M), narrow (6.7M), hasyv2 (13.3M)
  - tau=6.0, v_threshold=0.15 — proven healthy FR settings
  - grad_clip=0.5, min_lr=1e-4 — stability guards
  - Strict VRAM guard: <7GB target

Usage:
  python cuda/train_hasyv2.py --quick_test    # tiny model, 5 epoch, 10K data
  python cuda/train_hasyv2.py --full           # narrow model, 50 epoch, 151K data
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
    meta_spikeformer_hasyv2_tiny,
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
    parser = argparse.ArgumentParser(description='Meta-SpikeFormer HASYv2 (CUDA RTX 4060)')

    # Model
    parser.add_argument('--model_size', type=str, default='narrow',
                        choices=['tiny', 'narrow', 'hasyv2'])
    parser.add_argument('--T', type=int, default=3, help='SNN time steps')
    parser.add_argument('--tau', type=float, default=6.0, help='LIF time constant (v2: 6.0 for slower FR decline)')
    parser.add_argument('--v_threshold', type=float, default=0.15,
                        help='LIF threshold (v2: 0.15=much easier firing)')
    parser.add_argument('--drop_rate', type=float, default=0.05)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='0=use all available (151K)')
    parser.add_argument('--max_val_samples', type=int, default=0)

    # Spike regularization (prevent dead neurons / FR collapse)
    parser.add_argument('--lambda_sd', type=float, default=0.5,
                        help='Spike density reg weight (0=disabled, v5: 0.5)')
    parser.add_argument('--lambda_mp', type=float, default=0.02,
                        help='Membrane potential reg weight (0=disabled, v5: 0.02)')
    parser.add_argument('--target_fr_min', type=float, default=0.10,
                        help='Min target firing rate')

    # Dev preset (tiny model, small data, rapid iteration ~1min/epoch)
    parser.add_argument('--dev', action='store_true',
                        help='Dev: tiny model, 10 epochs, 5K samples, bs=32')

    # Quick test preset (tiny model, moderate data)
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test: tiny model, 10 epochs, 10K samples')

    # Full training preset (narrow model, full data)
    parser.add_argument('--full', action='store_true',
                        help='Full training: narrow model (6.7M), full 151K data')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='CUDA: 4 workers for data loading')
    parser.add_argument('--device', type=str, default='cuda')

    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Stop if no val improvement for N epochs (0=disable)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.1,
                        help='Minimum val acc improvement to reset patience')

    # FR guard
    parser.add_argument('--fr_critical', type=float, default=0.03,
                        help='Auto-stop if FR drops below this (default 3%%)')

    # Logging
    parser.add_argument('--log_csv', type=str, default='./logs/hasyv2_cuda_narrow_v3.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='')

    # Data
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    # Apply dev preset
    if args.dev:
        args.model_size = 'tiny'
        args.T = 2
        args.epochs = 10
        args.batch_size = 32
        args.max_train_samples = 5000
        args.max_val_samples = 2000
        args.warmup_epochs = 2
        args.early_stop_patience = 0
        args.save_every = 0
        args.lambda_sd = 0.5
        args.lambda_mp = 0.02
        args.log_csv = './logs/hasyv2_dev.csv'
        print("🔧 Dev: tiny model, 10 epochs, 5K train, bs=32, reg v5")

    # Apply quick_test preset
    if args.quick_test:
        args.model_size = 'tiny'
        args.T = 2
        args.epochs = 10
        args.batch_size = 16
        args.max_train_samples = 10000
        args.max_val_samples = 2000
        args.warmup_epochs = 2
        args.early_stop_patience = 0
        args.save_every = 0
        args.lambda_sd = 0.5
        args.lambda_mp = 0.02
        args.log_csv = './logs/hasyv2_tiny_test.csv'
        print("🧪 Quick test: tiny model, 10 epochs, 10K train, reg v5")

    # Apply full training preset
    if args.full:
        args.model_size = 'narrow'
        args.T = 3
        args.epochs = 50
        args.batch_size = 16
        args.max_train_samples = 0      # all 151K
        args.max_val_samples = 0
        args.warmup_epochs = 5
        args.early_stop_patience = 15
        args.save_every = 5
        args.lambda_sd = 0.5
        args.lambda_mp = 0.02
        args.log_csv = './logs/hasyv2_cuda_narrow_v5.csv'
        print("🚀 Full training: narrow model (6.7M), 50 epochs, 151K data, reg v5")

    # Compute effective batch
    args.eff_batch = args.batch_size
    print(f"🚀 HASYv2 CUDA: {args.model_size} model, {args.epochs} epochs, "
          f"batch={args.batch_size}, T={args.T}, tau={args.tau}, v_th={args.v_threshold}, "
          f"min_lr={args.min_lr}, early_stop={args.early_stop_patience}, "
          f"reg=(sd={args.lambda_sd}, mp={args.lambda_mp})")

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
    print(f"Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram_total:.1f} GB")

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
    model_map = {
        'tiny': meta_spikeformer_hasyv2_tiny,
        'narrow': meta_spikeformer_hasyv2_narrow,
        'hasyv2': meta_spikeformer_hasyv2,
    }
    model_kwargs = dict(
        T=args.T, tau=args.tau, v_threshold=args.v_threshold,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        use_groupnorm=True,
    )
    model = model_map[args.model_size](**model_kwargs).to(device)
    functional.set_step_mode(model, 'm')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = estimate_memory(model, args.batch_size, args.T, input_shape=(1, 32, 32))
    print(f"Model: {args.model_size}, {n_params/1e6:.2f}M params, T={args.T}")
    print(f"Estimated memory: ~{mem['total_mb']:.0f}MB "
          f"(params:{mem['params_mb']:.0f} grads:{mem['grads_mb']:.0f} "
          f"optim:{mem['optim_mb']:.0f} act:{mem['peak_act_mb']:.0f})")

    # ---- VRAM check (strict <7GB) ----
    vram_used = torch.cuda.memory_allocated() / 1e6
    vram_limit_mb = 7000
    print(f"Current VRAM: {vram_used:.0f}MB (limit: {vram_limit_mb}MB)")
    if vram_used > vram_limit_mb * 0.5:
        print(f"⚠️ VRAM already high before training, check other processes")

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
        csv_file.write('epoch,batch,loss,acc,lr,grad_norm,fr,ce_loss,sd_loss,mp_loss\n')

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
            'ce_loss': ce_loss, 'sd_loss': sd_loss, 'mp_loss': mp_loss})

    # ---- Resume ----
    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ---- Early stopping state ----
    no_improve_count = 0
    best_model_path = str(save_dir / f'hasyv2_{args.model_size}_best.pt')

    # ---- Training ----
    print(f"\n=== Training ({args.epochs} epochs, {len(train_loader)} batches/epoch, "
          f"early_stop={args.early_stop_patience}) ===")
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            grad_clip=args.grad_clip, fr_monitor=fr_monitor, dry_run=False,
            batch_logger=batch_logger if csv_file else None,
            grad_accum_steps=1,
            lambda_sd=args.lambda_sd, lambda_mp=args.lambda_mp,
            target_fr_min=args.target_fr_min)

        # Eval
        val_metrics = evaluate(model, val_loader, criterion, device,
                               fr_monitor=fr_monitor)

        epoch_time = time.time() - t0

        # FR + SOPs
        avg_fr = fr_monitor.get_avg_firing_rate()
        sops = estimate_sops(model, avg_fr, T=args.T, input_shape=(1, 32, 32))
        health = fr_monitor.check_health()
        fr_monitor.clear()

        # VRAM check
        vram_mb = torch.cuda.memory_allocated() / 1e6
        vram_warn = ""
        if vram_mb > vram_limit_mb:
            vram_warn = f" ⚠️VRAM {vram_mb:.0f}MB > {vram_limit_mb}MB LIMIT!"

        # Checkpoint
        is_best = val_metrics['acc'] > best_acc + args.early_stop_min_delta / 100.0
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
        health_str = f" [{health}]" if health else ""
        fr_health_str = ""
        if avg_fr < args.fr_critical:
            fr_health_str = " 💀FR CRITICAL!"
        elif avg_fr < 0.05:
            fr_health_str = " ⚠️FR LOW"

        reg_str = ""
        if args.lambda_sd > 0 or args.lambda_mp > 0:
            reg_str = (f" | Reg(CE:{train_metrics.get('ce_loss', 0):.3f} "
                       f"SD:{train_metrics.get('sd_loss', 0):.4f} "
                       f"MP:{train_metrics.get('mp_loss', 0):.4f})")
        print(f"Epoch {epoch+1:3d}/{args.epochs} | LR: {current_lr:.6f} | "
              f"Train: {train_metrics['loss']:.4f} {train_metrics['acc']:.2f}% | "
              f"Val: {val_metrics['loss']:.4f} {val_metrics['acc']:.2f}% | "
              f"FR: {avg_fr:.4f}{fr_health_str}{health_str} | SOPs: {sops:.2f}M | "
              f"VRAM: {vram_mb:.0f}MB{vram_warn} | Time: {epoch_time:.0f}s"
              f"{reg_str} {'*' if is_best else ''}")

        # Early stopping
        if is_best:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
            print(f"⏹ Early stop: no val improvement for {no_improve_count} epochs")
            break

        # FR auto-stop
        if avg_fr < args.fr_critical and epoch > args.warmup_epochs:
            print(f"💀 FR collapsed to {avg_fr:.4f} (limit={args.fr_critical}) — restoring best model")
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            break

        # VRAM auto-stop
        if vram_mb > vram_limit_mb:
            print(f"🛑 VRAM exceeded {vram_limit_mb}MB limit — stopping for safety")
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
        if batch_records:
            analysis = analyze_batch_records(batch_records)
            analysis_path = args.log_csv.replace('.csv', '_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Logs: {args.log_csv}, {analysis_path}")
            print(f"\n=== Analysis ===")
            print(f"Total batches: {analysis.get('total_batches', 'N/A')}")
            print(f"Loss: {analysis.get('loss_start', 0):.4f} → {analysis.get('loss_end', 0):.4f}")
            print(f"FR: {analysis.get('fr_mean', 0):.4f} ± {analysis.get('fr_std', 0):.4f}")


if __name__ == '__main__':
    main()
