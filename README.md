# MetaSpikeFormer

**Spike-Driven Transformer V2** for image classification.

Based on *"Spike-driven Transformer V2: Meta Spiking Neural Network Architecture"* (ICLR 2024).

PyTorch + SpikingJelly. Tested on Mac M1 Pro (MPS) and ready for NVIDIA RTX 4060 (CUDA).

## Project Structure

```
MetaSpikeFormer/
├── model.py                  # Core SNN architecture
├── dataset.py                # CIFAR-100 data pipeline
├── dataset_hasyv2.py         # HASYv2 data pipeline
├── train.py                  # Base training engine
├── mac/                      # Apple Silicon MPS
│   ├── train.py              # CIFAR-100 training
│   └── train_hasyv2.py       # HASYv2 training
├── cuda/                     # NVIDIA CUDA (RTX 4060)
│   └── train.py              # CIFAR-100 training
├── checkpoints/              # Model checkpoints
├── logs/                     # Training CSV logs
└── data/                     # Datasets
```

## Model Variants

| Model | Params | Blocks | Max Dim | Use Case |
|-------|--------|--------|---------|----------|
| tiny | 1.7M | 6 | 256 | Quick experiments, Mac |
| cifar100 | 11.6M | 10 | 512 | CIFAR-100 full training (CUDA) |
| hasyv2_narrow | 6.7M | 10 | 384 | HASYv2 Mac MPS |
| hasyv2 | 13.3M | 12 | 512 | HASYv2 full training (CUDA) |

## Critical Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `v_threshold` | **0.3** | Default 1.0 → 3% FR (dead neurons); 0.3 → 25% FR |
| `batch_size` | **16** | Small batch → gradient noise → better SNN generalization |
| `lr` | **1e-3** | Higher LR needed for SNN surrogate gradient |
| `drop_rate` | **0.1** | Regularization to prevent overfitting |
| `tau` | **2.0** | Faster decay → higher FR |

## Experiment Results (Mac M1 Pro)

### CIFAR-100 (100 classes, RGB 32×32)

| Run | Model | Data | Epochs | Batch | Best Val | Key |
|-----|-------|------|--------|-------|----------|-----|
| 1 | tiny | 5K | 5 | 16 | 1.70% | v_th=1.0, FR dead |
| 2 | tiny | 5K | 5 | 16 | 6.35% | v_th=0.3 fix |
| 3 | tiny | 10K | 6 | 32 | 1.75% | batch too large |
| 4 | tiny | 20K | 8 | 32 | 3.56% | lr too low |
| **5** | **tiny** | **25K** | **12** | **16** | **29.38%** | 🏆 best config |

**Best command:**
```bash
python mac/train.py --model_size tiny --epochs 12 --batch_size 16 \
  --v_threshold 0.3 --lr 1e-3 --drop_rate 0.1 --warmup_epochs 3 \
  --max_train_samples 25000
```

### HASYv2 (369 math symbols, grayscale 32×32)

| Run | Model | Data | Epochs | Batch | Best Val | FR trend | Key |
|-----|-------|------|--------|-------|----------|----------|-----|
| quick | tiny | 10K | 3 | 16 | 0.70% | 0.25→0.26 | model too small |
| **half** | **tiny** | **75K** | **20** | **16** | **46.83%** | 0.25→0.034 | 🏆 best on Mac |
| full (plan) | narrow | 75K | 30 | 16 | TBD | — | efficient 6.7M |
| full (plan) | hasyv2 | 151K | 40 | 16 | TBD | — | CUDA only |

**Best command:**
```bash
python mac/train_hasyv2.py --preset half
# tiny model, 75K samples, 20 epochs, T=3
```

## Quick Start

### Mac MPS
```bash
# CIFAR-100
python mac/train.py --preset full

# HASYv2
python mac/train_hasyv2.py --preset half    # narrow model, 75K data
python mac/train_hasyv2.py --preset dryrun  # verify pipeline
```

### CUDA (NVIDIA RTX 4060)
```bash
# CIFAR-100 quick
python cuda/train.py --preset quick

# CIFAR-100 full
python cuda/train.py --preset standard
```

## Architecture

- **SDSA**: Spike-Driven Self-Attention with linear O=Q·(KᵀV) attention
- **SPS**: Spike Patch Splitting for hierarchical down-sampling
- **LIF Neurons**: ATan surrogate gradient, configurable tau/threshold
- **GroupNorm**: Batch-independent normalization for small-batch stability
- **Multi-step**: T=3-4 time steps for temporal spike integration

## Requirements

- Python 3.11, PyTorch 2.5+, SpikingJelly 0.0.0.0.14
- torchvision, einops, timm
