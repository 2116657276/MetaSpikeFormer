# MetaSpikeFormer

**Spike-Driven Transformer V2** for image classification on neuromorphic hardware.

Based on *"Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-Generation Neuromorphic Chips"* (ICLR 2024).

PyTorch + SpikingJelly. Tested on **Mac M1 Pro (MPS)** and **NVIDIA RTX 4060 Laptop (CUDA)**.

---

## 项目结构

```
MetaSpikeFormer/
├── model.py                  # SNN 架构 (SDSA + SPS + PLIF/LIF)
├── train.py                  # 训练引擎 (loop/eval/FR/V-reg/checkpoint)
├── dataset.py                # CIFAR-100 数据管线
├── dataset_hasyv2.py         # HASYv2 数据管线 (user-based split)
├── cuda/
│   ├── train.py              # CIFAR-100 CUDA 训练
│   └── train_hasyv2.py       # HASYv2 CUDA 训练
├── mac/
│   ├── train.py              # CIFAR-100 MPS 训练
│   └── train_hasyv2.py       # HASYv2 MPS 训练
├── checkpoints/              # 模型权重
├── logs/                     # 训练 CSV 日志
└── data/                     # 数据集
```

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.6+ |
| SpikingJelly | 0.0.0.0.14+ |
| torchvision | 0.21+ |
| CUDA (可选) | 12.4 |

```bash
pip install torch torchvision spikingjelly
```

---

## 模型变体

| 模型 | 参数 | 层数 | 通道维度 | 神经元 | 用途 |
|------|------|------|---------|:---:|------|
| **narrow** | 6.67M | 10 blocks, 4 stages | [48, 96, 192, 384] | PLIF | HASYv2 主力 |
| **shallow** | 2.70M | 7 blocks, 3 stages | [48, 128, 256] | PLIF | 快速实验 |
| hasyv2 | 13.3M | 12 blocks, 4 stages | [64, 128, 256, 512] | PLIF | 全量 CUDA |
| tiny | 0.80M | 5 blocks, 4 stages | [24, 48, 96, 192] | LIF | 快速验证 |
| cifar100 | 11.6M | 10 blocks, 4 stages | [64, 128, 256, 512] | LIF | CIFAR-100 |

---

## 架构详解

### 关键组件

- **PLIF (Parametric LIF)**：tau 和 v_threshold 逐层可学习。网络在训练中自动调节放电特性，解决固定 LIF 的 FR 崩溃问题。
- **SDSA (Spike-Driven Self-Attention)**：Q/K/V 经过 Linear→Norm→PLIF 产生二值脉冲，使用线性注意力 O=Q·(KᵀV) 避免 O(N²) softmax。
- **SPS (Spike Patch Splitting)**：步长 2 的 3×3 卷积 + BN + PLIF，分层下采样。
- **GroupNorm**：batch 无关归一化，适应小 batch 训练。
- **V-based Regularization**：仅惩罚负膜电位 (relu(-v)²)，不干预正常放电，梯度通过膜电位直接传导。

### narrow 架构

```
Patch Embed (conv 3×3) → BN → PLIF
Stage 1: [48-dim] × 2 blocks
         ↓ SPS (48→96)
Stage 2: [96-dim] × 2 blocks
         ↓ SPS (96→192)
Stage 3: [192-dim] × 4 blocks
         ↓ SPS (192→384)
Stage 4: [384-dim] × 2 blocks
         → GAP → Linear(369)
Total: 10 blocks, 64 PLIF nodes, 6.67M params
```

---

## 数据集

### HASYv2

- **369 类** LaTeX 数学符号，~168K 灰度 32×32 图像
- **User-based split**：训练/验证集零用户重叠（`val_user_frac=0.15`）
- 增强：RandomCrop(32, pad=2) + RandomAffine(±15°, 15%平移, 0.85-1.15缩放)
- 归一化：mean=0.9372, std=0.2310

### CIFAR-100

- 100 类 RGB 32×32 图像，50K 训练 / 10K 测试

---

## 快速入门

### CUDA (RTX 4060, 8GB VRAM)

```bash
# 开发测试 (shallow 2.7M, 10K, 10ep, ~5 min)
python cuda/train_hasyv2.py --quick

# 半量训练 (narrow 6.7M, 75K, 25ep, ~9 h)
python cuda/train_hasyv2.py --half

# 全量训练 (narrow 6.7M, 151K, 25ep)
python cuda/train_hasyv2.py --narrow

# CIFAR-100
python cuda/train.py --preset standard
```

### Mac MPS

```bash
# 快速测试
python mac/train_hasyv2.py --preset quick

# 正式训练
python mac/train_hasyv2.py --preset half
```

---

## 实验结果

### HASYv2

#### CUDA — RTX 4060 (narrow PLIF, 最佳)

| 参数 | 值 | | 参数 | 值 |
|------|------|------|------|------|
| **Best Val Acc** | **77.19%** (E23) | | lr / min_lr | 1e-3 / 1e-5 |
| 模型 / 参数 / 神经元 | narrow / 6.67M / 64 PLIF | | weight_decay | 0.02 |
| T / tau / v_th | 3 / 2.0 / 0.25 | | warmup / grad_clip | 3 / 1.0 |
| bs / epochs | 16 / 25 | | V-reg (λ) | 0.3 / 0.1 |
| 训练数据 / Val 数据 | 75K / ~30K (user-split) | | VRAM / GPU / 时长 | 3.9 GB / 63°C / 9.7 h |
| FR 终值 | 0.278 (E13-E25 零衰减) | | 权重 | `checkpoints/hasyv2_narrow_best.pt` |

**代表性 Epoch：**

| Ep | 1 | 3 | 10 | 15 | 20 | 23 🏆 | 25 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FR | 0.224 | 0.416 | 0.333 | 0.305 | 0.285 | 0.278 | 0.277 |
| Val% | 2.3 | 47.1 | 68.9 | 73.1 | 76.5 | **77.19** | 77.06 |

**FR 走势：** 峰值 0.42(E3) → 稳态 0.28(E13-E25)，连续 12 epoch 无衰减，PLIF 完全收敛。

**复现命令：**

```bash
python cuda/train_hasyv2.py \
  --model_size narrow --T 3 --batch_size 16 --epochs 25 \
  --v_threshold 0.25 --tau 2.0 --drop_rate 0.1 \
  --lr 1e-3 --min_lr 1e-5 --warmup_epochs 3 --grad_clip 1.0 \
  --weight_decay 0.02 --max_train_samples 75000 \
  --lambda_v 0.3 --lambda_vneg 0.1 \
  --early_stop_patience 10 --save_every 5
```

#### Mac — M1 Pro (narrow 固定 LIF, 最佳)

| 参数 | 值 | | 参数 | 值 |
|------|------|------|------|------|
| **Best Val Acc** | **78.69%** (E25) | | lr / min_lr | 1e-3 / 1e-5 |
| 模型 / 参数 | narrow / 6.67M (固定 LIF) | | weight_decay | 0.02 |
| T / tau / v_th | 3 / 2.0 / 0.25 | | warmup / grad_clip | 3 / 1.0 |
| bs / epochs | 16 / 25 | | 训练数据 / Val | 75K / ~15K |
| FR 终值 | ~0.11 | | | |

#### CUDA — RTX 4060 (shallow PLIF, 次佳)

| 参数 | 值 | | 参数 | 值 |
|------|------|------|------|------|
| **Best Val Acc** | **71.32%** (E25) | | T / tau / v_th | 3 / 2.0 / 0.20 |
| 模型 / 参数 | shallow / 2.70M / 45 PLIF | | 训练数据 | 75K |
| FR 终值 | 0.223 (完全稳定) | | VRAM / 时长 | 5.2 GB / 8.2 h |

#### CUDA vs Mac 对比

| 维度 | CUDA | Mac |
|------|:---:|:---:|
| Best Val | 77.19% | 78.69% |
| 神经元类型 | PLIF (可学习) | 固定 LIF |
| Val 验证集 | ~30K (全量) | ~15K (max_val_samples=15000) |
| FR 稳态 | 0.278 | 0.11 |
| 差异分析 | 验证集规模差异（CUDA 更严格） + 随机种子波动 | — |

#### 历史实验总结

| 实验 | 模型 | 参数 | Val% | FR | 结果 |
|------|------|------|:---:|------|:---:|
| CUDA v7 | narrow PLIF | 6.67M | **77.19** | 0.28 | ✅ 成功 |
| Mac v2 | narrow LIF | 6.67M | **78.69** | 0.11 | ✅ 成功 |
| CUDA v7 | shallow PLIF | 2.70M | 71.32 | 0.22 | ✅ 成功 |
| CUDA v3 | narrow LIF | 6.67M | — | 0.04 | 💀 FR 崩溃 |
| CUDA v4 | narrow LIF+reg | 6.67M | 53.65 | 0.05 | 💀 .detach() 断梯度 |
| CUDA v5 | narrow LIF+reg | 6.67M | 28.36 | 0.06 | 💀 reg ÷64 稀释 |
| CUDA v6 | narrow LIF+reg | 6.67M | 28.68 | 0.04 | 💀 ATan 死角 |

---

## 超参数指南

| 参数 | 推荐值 | 说明 |
|------|------|------|
| `v_threshold` | 0.20-0.25 | 放电阈值。太高→FR 低，太低→FR 崩溃 |
| `T` | 3 | 灰度图 3 步足够穿过 10 blocks；T=2 深层信号不足 |
| `tau` | 2.0 | 膜电位衰减常数 |
| `batch_size` | 16 | 小 batch → 梯度噪声 → SNN 泛化更好 |
| `lr` | 1e-3 | SNN 需要高于传统网络的初始学习率 |
| `min_lr` | 1e-5 | 末期冻结学习，防止扰动已稳神经元 |
| `weight_decay` | 0.02 | 低于常规值 (0.05)，保护 LIF 输入信号 |
| `warmup_epochs` | 3 | 让 PLIF 有时间自调 tau/v_th |
| `λ_v` / `λ_vneg` | 0.3 / 0.1 | V-based 轻量保护，0=禁用 |

---

## 许可证

基于 ICLR 2024 *"Spike-driven Transformer V2"* 开源实现。HASYv2 数据集来自 [Martin Thoma](https://github.com/MartinThoma/hasy)。
