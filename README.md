# MetaSpikeFormer

**Spike-Driven Transformer V2** for CIFAR-100 classification.

Based on *"Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-Generation Neuromorphic Chips"* (ICLR 2024).

Built with PyTorch + SpikingJelly.

## Project Structure

```
MetaSpikeFormer/
├── model.py              # Core SNN architecture (shared)
├── dataset.py            # CIFAR-100 data pipeline (shared)
├── train.py              # Base training engine (shared)
├── mac/                  # Apple Silicon MPS training
│   └── train.py          # MPS-optimized training script
├── cuda/                 # NVIDIA CUDA training (RTX 4060)
│   └── train.py          # CUDA-optimized training script
├── checkpoints/          # Saved model checkpoints (.gitignored)
├── logs/                 # Training CSV logs (.gitignored)
└── data/                 # CIFAR-100 dataset (.gitignored)
```

## Quick Start

### Mac (Apple Silicon MPS)
```bash
# Quick test (3 epochs, micro model, 2500 samples)
python mac/train.py --preset quick

# Full training (36 epochs, tiny model, all data)
python mac/train.py --preset full

# Custom
python mac/train.py --model_size cifar100 --epochs 50 --batch_size 16
```

### CUDA (NVIDIA RTX 4060 / WSL)
```bash
# Quick test (30 epochs, tiny model)
python cuda/train.py --preset quick

# Standard training (200 epochs, cifar100 model)
python cuda/train.py --preset standard

# Custom
python cuda/train.py --model_size cifar100 --epochs 300 --batch_size 128
```

## Model Variants

| Variant | Params | Memory (train) | Use Case |
|---------|--------|---------------|----------|
| micro | 0.29M | ~5MB | Pipeline verification |
| tiny | 1.73M | ~21MB | Quick experiments |
| cifar100 | 11.61M | ~140MB | Full training |

## Critical Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `v_threshold` | **0.3** | Lower→higher firing rate. Default 1.0 gives only 3% FR, model can't learn |
| `batch_size` | 16 (MPS) / 64 (CUDA) | Small batch→gradient noise→better SNN generalization |
| `lr` | 1e-3 | Higher LR needed for SNN surrogate gradient |
| `drop_rate` | 0.1 | Regularization to prevent overfitting |
| `warmup_epochs` | 3-10 | Essential for SNN training stability |
| `tau` | 2.0 | LIF time constant (lower→faster decay→higher FR) |

## Training Results (Mac M1 Pro)

| Run | Model | Data | Epochs | Best Val Acc | Key Changes |
|-----|-------|------|--------|-------------|-------------|
| 1 | micro | 5K | 5 | 1.70% | v_th=1.0 (FR=3%) |
| 2 | micro | 5K | 5 | 6.35% | v_th=0.3 |
| 3 | micro | 10K | 6 | 1.75% | batch=32 (too large) |
| 4 | tiny | 20K | 8 | 3.56% | lr=5e-4, wd=0.1 |
| **5** | **tiny** | **25K** | **12** | **29.38%** | batch=16, lr=1e-3, dropout=0.1 |
| 6* | tiny | 50K | 8/36 | 29.32% | Full data, ongoing |

**Best configuration (Run 5):**
```bash
python mac/train.py --model_size tiny --epochs 12 --batch_size 16 \
  --v_threshold 0.3 --tau 2.0 --lr 1e-3 --drop_rate 0.1 \
  --warmup_epochs 3 --max_train_samples 25000
```

## Architecture

- **Spike-Driven Self-Attention (SDSA)**: Linear attention O=Q·(KᵀV) with binary spike tensors
- **Spike Patch Splitting (SPS)**: Hierarchical down-sampling with strided conv + LIF
- **LIF Neurons**: ATan surrogate gradient, v_reset=0.0, configurable tau/threshold
- **GroupNorm**: Batch-independent normalization for small-batch stability
- **Multi-step mode**: T=4 time steps for temporal integration

## Environment

- Python 3.11
- PyTorch 2.5+
- SpikingJelly 0.0.0.0.14
- torchvision, einops, timm

## License

Research project — for academic use.
