#!/usr/bin/env python3
"""
MetaSpikeFormer — CUDA (RTX 4060) Training Script
Optimized for NVIDIA RTX 4060 (8GB VRAM) on WSL/Ubuntu.

Key CUDA optimizations:
  - batch_size=64 (larger batch, GPU has enough VRAM)
  - num_workers=4 (parallel data loading via CPU)
  - Full CIFAR-100 dataset
  - Automatic Mixed Precision (AMP) for faster training
  - cifar100 model (11.6M params) for best accuracy

Usage:
  # Quick test (tiny model, 12 epochs)
  python cuda/train.py --preset quick

  # Standard training (cifar100 model, 200 epochs)
  python cuda/train.py --preset standard

  # Full training with custom params
  python cuda/train.py --model_size cifar100 --epochs 300 --batch_size 128
"""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from model import meta_spikeformer_micro, meta_spikeformer_tiny, meta_spikeformer_cifar100
from dataset import build_cifar100
from train import train_one_epoch, evaluate, FiringRateMonitor, estimate_sops
from train import estimate_memory, save_checkpoint, load_checkpoint
from train import WarmupCosineScheduler, analyze_batch_records
from spikingjelly.activation_based import functional, neuron


def get_args():
    parser = argparse.ArgumentParser(description='Meta-SpikeFormer (CUDA RTX 4060)')

    # Presets
    parser.add_argument('--preset', type=str, default='',
                        choices=['quick', 'standard', ''],
                        help='quick=test, standard=full training')

    # Model
    parser.add_argument('--model_size', type=str, default='cifar100',
                        choices=['micro', 'tiny', 'cifar100'])
    parser.add_argument('--T', type=int, default=4, help='SNN time steps')
    parser.add_argument('--tau', type=float, default=2.0, help='LIF time constant')
    parser.add_argument('--v_threshold', type=float, default=0.3,
                        help='LIF threshold (0.3=critical for healthy FR)')
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='0=use all CIFAR-100 (50000)')
    parser.add_argument('--max_val_samples', type=int, default=0)

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='CUDA: use 4+ for data loading')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (recommended)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP')

    # Logging
    parser.add_argument('--log_csv', type=str, default='./logs/cuda_training.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()

    # Apply presets
    if args.preset == 'quick':
        args.model_size = 'tiny'
        args.epochs = 30
        args.batch_size = 64
        args.warmup_epochs = 5
        print("🚀 Preset 'quick': tiny model, 30 epochs, full data")
    elif args.preset == 'standard':
        args.model_size = 'cifar100'
        args.epochs = 200
        args.batch_size = 64
        args.warmup_epochs = 10
        print("🚀 Preset 'standard': cifar100 model, 200 epochs, full data")

    return args


def main():
    args = get_args()

    # Device
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"  AMP: {'enabled' if args.use_amp and not args.no_amp else 'disabled'}")

    # Data
    print("Loading CIFAR-100...")
    train_loader, val_loader, num_classes = build_cifar100(
        batch_size=args.batch_size, num_workers=args.num_workers, T=args.T)

    from torch.utils.data import Subset, DataLoader
    if args.max_train_samples > 0:
        n = min(args.max_train_samples, len(train_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(n)),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f"  Train subset: {n} samples")
    if args.max_val_samples > 0:
        n = min(args.max_val_samples, len(val_loader.dataset))
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(n)),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        print(f"  Val subset: {n} samples")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print("Building model...")
    model_kwargs = dict(T=args.T, tau=args.tau, v_threshold=args.v_threshold,
                        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                        use_groupnorm=True)
    model_map = {'micro': meta_spikeformer_micro, 'tiny': meta_spikeformer_tiny,
                 'cifar100': meta_spikeformer_cifar100}
    model = model_map[args.model_size](**model_kwargs).to(device)
    functional.set_step_mode(model, 'm')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = estimate_memory(model, args.batch_size, args.T)
    print(f"Model: {args.model_size}, {n_params/1e6:.2f}M params, T={args.T}")
    print(f"Estimated memory: ~{mem['total_mb']:.0f}MB")

    # AMP scaler
    use_amp = args.use_amp and not args.no_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP scaler initialized")

    # Monitor
    fr_monitor = FiringRateMonitor(model)
    print(f"FR monitor: {len(fr_monitor.hooks)} LIF nodes")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs,
        base_lr=args.lr, min_lr=args.min_lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Save dir
    from pathlib import Path
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # CSV logging
    csv_file = open(args.log_csv, 'w') if args.log_csv else None
    batch_records = []
    if csv_file:
        csv_file.write('epoch,batch,loss,acc,lr,grad_norm,fr\n')

    def batch_logger(epoch, batch_idx, loss, acc, lr, grad_norm, fr):
        if csv_file:
            csv_file.write(f'{epoch},{batch_idx},{loss:.6f},{acc:.4f},{lr:.8f},{grad_norm:.4f},{fr:.6f}\n')
        batch_records.append({'epoch': epoch, 'batch': batch_idx, 'loss': loss,
                              'acc': acc, 'lr': lr, 'grad_norm': grad_norm, 'fr': fr})

    # Resume
    import time, json
    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # Training loop
    print(f"\n=== Training ({args.epochs} epochs, batch={args.batch_size}) ===")
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            grad_clip=args.grad_clip, fr_monitor=fr_monitor, dry_run=False,
            batch_logger=batch_logger if csv_file else None)

        val_metrics = evaluate(model, val_loader, criterion, device, fr_monitor=fr_monitor)
        epoch_time = time.time() - t0

        avg_fr = fr_monitor.get_avg_firing_rate()
        sops = estimate_sops(model, avg_fr, T=args.T)
        fr_monitor.clear()

        is_best = val_metrics['acc'] > best_acc
        if is_best:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, val_metrics,
                          str(Path(args.save_dir) / f'{args.model_size}_best_cuda.pt'))
            print(f"  → Best saved ({best_acc:.2f}%)")

        print(f"Epoch {epoch+1:3d}/{args.epochs} | LR: {current_lr:.6f} | "
              f"Train: {train_metrics['loss']:.4f} {train_metrics['acc']:.2f}% | "
              f"Val: {val_metrics['loss']:.4f} {val_metrics['acc']:.2f}% | "
              f"FR: {avg_fr:.4f} | SOPs: {sops:.2f}M | Time: {epoch_time:.0f}s {'*' if is_best else ''}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, val_metrics,
                          str(Path(args.save_dir) / f'{args.model_size}_epoch{epoch+1}_cuda.pt'))

    print(f"\n=== Done ===")
    print(f"Best Val Acc: {best_acc:.2f}%")

    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_acc, {},
                  str(Path(args.save_dir) / f'{args.model_size}_final_cuda.pt'))

    if csv_file:
        csv_file.close()
        analysis = analyze_batch_records(batch_records)
        analysis_path = args.log_csv.replace('.csv', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Logs: {args.log_csv}, {analysis_path}")


if __name__ == '__main__':
    main()
