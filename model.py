"""
Meta-SpikeFormer: Spike-driven Transformer V2

Reference:
  "Spike-driven Transformer V2: Meta Spiking Neural Network Architecture
   Inspiring the Design of Next-Generation Neuromorphic Chips" (ICLR 2024)

Core features:
  - Binary spike-driven Q/K/V via LIF neurons (ATan surrogate, v_reset=0.0).
  - Spike-Driven Self-Attention (SDSA) using linear (kernel) attention.
  - Spike Patch Splitting (SPS) for hierarchical down-sampling.
  - Multi-step mode: set_step_mode(model, 'm') for T timesteps.

Improvements over v1:
  - Fixed SDSA attention scaling (removed over-aggressive /N division).
  - Replaced BatchNorm1d with LayerNorm for small-batch stability.
  - Added GroupNorm option for token-wise normalization.
  - Configurable LIF parameters (tau, v_threshold).
  - Micro / Tiny / CIFAR100 model variants.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, surrogate, functional, layer


# ---------------------------------------------------------------------------
#  Helper: build a LIF node with the paper-recommended settings
# ---------------------------------------------------------------------------

def _lif(
    step_mode: str = 'm',
    tau: float = 2.0,
    v_threshold: float = 1.0,
) -> neuron.LIFNode:
    """Return a LIFNode with ATan surrogate gradient and v_reset=0.0."""
    return neuron.LIFNode(
        surrogate_function=surrogate.ATan(),
        v_reset=0.0,
        step_mode=step_mode,
        tau=tau,
        v_threshold=v_threshold,
    )


# ---------------------------------------------------------------------------
#  Helper: Token-wise normalization (batch-independent, for small batches)
# ---------------------------------------------------------------------------

def _norm(dim: int, use_groupnorm: bool = True) -> nn.Module:
    """
    Return a normalization layer suitable for token sequences.
    LayerNorm: normalizes over the last dimension, batch-independent.
    GroupNorm: normalizes over groups of channels.
    Both avoid the small-batch instability of BatchNorm.
    """
    if use_groupnorm:
        num_groups = max(1, dim // 8)  # ensure at least 1 group
        # Ensure num_groups divides dim
        while num_groups > 1 and dim % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, dim)
    else:
        return nn.LayerNorm(dim)


# ---------------------------------------------------------------------------
#  Spike-Driven Self-Attention (SDSA)
# ---------------------------------------------------------------------------

class SDSA(nn.Module):
    """
    Spike-Driven Self-Attention.

    Q, K, V are obtained through Linear → Norm → LIF, producing **binary**
    spike tensors.  Uses linearised (kernel) attention:
        O = Q @ (Kᵀ @ V)
    which avoids the O(N²) softmax and replaces matmul with operations
    friendly to neuromorphic hardware.

    Args:
        dim: feature dimension.
        num_heads: number of attention heads.
        sr_ratio: spatial reduction ratio (for PVT-style reduction).
        attn_drop, proj_drop: dropout rates.
        tau, v_threshold: LIF neuron parameters.
        use_groupnorm: if True, use GroupNorm; else LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.use_groupnorm = use_groupnorm

        # Q / K / V projection chains
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.q_norm = _norm(dim, use_groupnorm)
        self.k_norm = _norm(dim, use_groupnorm)
        self.v_norm = _norm(dim, use_groupnorm)

        self.q_lif = _lif('m', tau, v_threshold)
        self.k_lif = _lif('m', tau, v_threshold)
        self.v_lif = _lif('m', tau, v_threshold)

        # Spatial reduction (optional, like PVT)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.BatchNorm2d(dim)
        else:
            self.sr = None

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_norm = _norm(dim, use_groupnorm)
        self.proj_lif = _lif('m', tau, v_threshold)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def _apply_qkv_proj(self, x: torch.Tensor, linear, norm, lif):
        """
        Apply Linear → Norm → LIF to a token tensor.

        Args:
            x: [T, B, N, C] or similar multi-step tensor.
        Returns:
            Binary spike tensor of same shape.
        """
        T, B, N, C = x.shape
        out = linear(x)  # Linear over last dim → [T, B, N, C]

        # Norm: for LayerNorm([C]), apply directly over last dim.
        # for GroupNorm(C), need [T*B*N, C] or [T*B, C, N].
        if self.use_groupnorm:
            out_gn = out.reshape(T * B, N, C).transpose(1, 2)  # [T*B, C, N]
            out_gn = norm(out_gn)                                # GroupNorm over C
            out = out_gn.transpose(1, 2).reshape(T, B, N, C)    # back to [T,B,N,C]
        else:
            out = norm(out)  # LayerNorm over last dim

        out = lif(out)  # Binary spikes
        return out

    def forward(self, x: torch.Tensor, H_spatial: int, W_spatial: int):
        """
        Args:
            x:  [T, B, N, dim]
            H_spatial, W_spatial: spatial resolution (needed for sr_ratio > 1).
        Returns:
            [T, B, N, dim]
        """
        T, B, N, C = x.shape

        # ---- Q, K, V projections ----
        q = self._apply_qkv_proj(x, self.q_linear, self.q_norm, self.q_lif)
        k = self._apply_qkv_proj(x, self.k_linear, self.k_norm, self.k_lif)
        v = self._apply_qkv_proj(x, self.v_linear, self.v_norm, self.v_lif)

        # ---- Spatial reduction on K, V (if sr_ratio > 1) ----
        if self.sr is not None:
            def _sr(t):
                t_sp = t.reshape(T * B, H_spatial, W_spatial, C).permute(0, 3, 1, 2)
                t_sp = self.sr(t_sp)
                t_sp = self.sr_norm(t_sp)
                _, C_, H_, W_ = t_sp.shape
                t_sp = t_sp.reshape(T, B, C_, H_ * W_).transpose(2, 3)
                return t_sp
            k = _sr(k)
            v = _sr(v)
            N_kv = k.shape[2]
        else:
            N_kv = N

        # ---- Multi-head reshape ----
        q = q.reshape(T, B, N, self.num_heads, self.head_dim)
        k = k.reshape(T, B, N_kv, self.num_heads, self.head_dim)
        v = v.reshape(T, B, N_kv, self.num_heads, self.head_dim)

        # ---- Spike-Driven (Linear) Attention ----
        # O = Q @ (Kᵀ @ V)  with proper scaling
        # KᵀV: [T, B, H, D, D] ← einsum('...n h d, ...n h e -> ...h d e', K, V)
        kv = torch.einsum('t b n h d, t b n h e -> t b h d e', k, v)  # [T,B,H,D,D]
        # Scale by 1/sqrt(D) — no /N division, LayerNorm after attention handles amplitude
        kv = kv * self.scale

        # Q @ (KᵀV): [T, B, N, H, D] @ [T, B, H, D, D] → [T, B, N, H, D]
        out = torch.einsum('t b n h d, t b h d e -> t b n h e', q, kv)

        # Back to [T, B, N, C]
        out = out.reshape(T, B, N, C)

        # ---- Output projection ----
        # Output LayerNorm for stability (normalizes attention output)
        if self.use_groupnorm:
            out_gn = out.reshape(T * B, N, C).transpose(1, 2)
            out_gn = self.proj_norm(out_gn)
            out = out_gn.transpose(1, 2).reshape(T, B, N, C)
        else:
            out = self.proj_norm(out)

        out = self.proj(out)                                         # [T, B, N, C]
        out = self.proj_lif(out)                                     # binary spikes
        out = self.attn_drop(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
#  Spike MLP
# ---------------------------------------------------------------------------

class SpikeMLP(nn.Module):
    """MLP with intermediate LIF activation (binary spikes)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        self.use_groupnorm = use_groupnorm
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = _norm(hidden_features, use_groupnorm)
        self.lif1 = _lif('m', tau, v_threshold)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2 = _norm(out_features, use_groupnorm)
        self.lif2 = _lif('m', tau, v_threshold)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, N, C]
        Returns:
            [T, B, N, out_features]
        """
        T, B, N, C = x.shape

        # FC1 → Norm → LIF
        o = self.fc1(x)  # [T, B, N, hidden]
        o = self._apply_norm(o, self.norm1)
        o = self.lif1(o)  # binary spikes
        o = self.drop(o)

        # FC2 → Norm → LIF
        o = self.fc2(o)  # [T, B, N, out]
        o = self._apply_norm(o, self.norm2)
        o = self.lif2(o)  # binary spikes
        o = self.drop(o)
        return o

    def _apply_norm(self, x, norm):
        if self.use_groupnorm:
            T, B, N, C = x.shape
            x = x.reshape(T * B, N, C).transpose(1, 2)  # [T*B, C, N]
            x = norm(x)
            x = x.transpose(1, 2).reshape(T, B, N, -1)    # [T, B, N, C]
        else:
            x = norm(x)  # LayerNorm over last dim
        return x


# ---------------------------------------------------------------------------
#  Meta-SpikeFormer Block  (SDSA + MLP)
# ---------------------------------------------------------------------------

class MetaSpikeBlock(nn.Module):
    """One Meta-SpikeFormer encoder block: SDSA → MLP, both with skip connections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sr_ratio: int = 1,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        self.use_groupnorm = use_groupnorm
        self.norm1 = _norm(dim, use_groupnorm)
        self.sdsa = SDSA(dim, num_heads, sr_ratio, attn_drop, drop,
                         tau=tau, v_threshold=v_threshold, use_groupnorm=use_groupnorm)
        self.norm2 = _norm(dim, use_groupnorm)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SpikeMLP(dim, mlp_hidden, dim, drop,
                            tau=tau, v_threshold=v_threshold, use_groupnorm=use_groupnorm)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x:   [T, B, N, C]
            H,W: spatial resolution
        Returns:
            [T, B, N, C]
        """
        T, B, N, C = x.shape

        # SDSA with pre-norm + residual
        if self.use_groupnorm:
            normed = x.reshape(T * B, N, C).transpose(1, 2)
            normed = self.norm1(normed)
            normed = normed.transpose(1, 2).reshape(T, B, N, C)
        else:
            normed = self.norm1(x)
        x = x + self.sdsa(normed, H, W)

        # MLP with pre-norm + residual
        if self.use_groupnorm:
            normed = x.reshape(T * B, N, C).transpose(1, 2)
            normed = self.norm2(normed)
            normed = normed.transpose(1, 2).reshape(T, B, N, C)
        else:
            normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


# ---------------------------------------------------------------------------
#  Spike Patch Splitting (SPS) — down-sampling module
# ---------------------------------------------------------------------------

class SPS(nn.Module):
    """
    Spike Patch Splitting.

    Reduces spatial resolution by 2× and doubles the channel dimension,
    using a strided convolution followed by BN → LIF to produce binary spikes.
    BatchNorm2d is kept here because spatial convolutions have large enough
    effective batch size (T*B*H*W samples per channel).
    """

    def __init__(self, in_dim: int, out_dim: int, tau: float = 2.0, v_threshold: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.lif = _lif('m', tau, v_threshold)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x:  [T, B, N, C]   (N == H*W)
        Returns:
            x:  [T, B, N//4, out_dim], H_out, W_out
        """
        T, B, N, C = x.shape
        H_out, W_out = H // 2, W // 2
        x_sp = x.reshape(T * B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [T*B, C, H, W]
        x_sp = self.conv(x_sp)
        x_sp = self.bn(x_sp)
        x_sp = x_sp.reshape(T, B, -1, H_out * W_out).permute(0, 1, 3, 2)   # [T, B, N', C']
        x_sp = self.lif(x_sp)                                                # binary spikes
        return x_sp, H_out, W_out


# ---------------------------------------------------------------------------
#  Full Meta-SpikeFormer
# ---------------------------------------------------------------------------

class MetaSpikeFormer(nn.Module):
    """
    Meta-SpikeFormer for CIFAR-100 classification.

    Hierarchical architecture with 4 stages, each consisting of an SPS
    down-sampling layer followed by multiple MetaSpikeBlocks.

    Args:
        img_size:        input image size (square).
        in_channels:     number of input channels (3 for RGB).
        num_classes:     output classes (100 for CIFAR-100).
        embed_dims:      channel dimensions for the 4 stages.
        depths:          number of blocks per stage.
        num_heads:       attention heads per stage.
        mlp_ratios:      MLP hidden expansion ratios.
        T:               SNN time steps.
        tau, v_threshold: LIF neuron parameters.
        drop_rate, attn_drop_rate:  dropout rates.
        use_groupnorm:   if True, use GroupNorm instead of LayerNorm.
    """

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dims: Tuple[int, ...] = (64, 128, 256, 512),
        depths: Tuple[int, ...]     = (2, 2, 4, 2),
        num_heads: Tuple[int, ...]  = (4, 8, 16, 32),
        mlp_ratios: Tuple[float, ...] = (4.0, 4.0, 4.0, 4.0),
        T: int = 4,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_groupnorm: bool = True,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_stages = len(embed_dims)
        self.use_groupnorm = use_groupnorm

        # ---- Patch Embedding (conv stem) ----
        self.patch_embed = nn.Conv2d(in_channels, embed_dims[0],
                                      kernel_size=3, stride=1, padding=1)
        self.patch_bn = nn.BatchNorm2d(embed_dims[0])
        self.patch_lif = _lif('m', tau, v_threshold)

        # ---- Stages ----
        self.stages = nn.ModuleList()
        self.sps_modules = nn.ModuleList()
        in_dim = embed_dims[0]

        for i in range(self.num_stages):
            # SPS down-sample (except first stage where we already have tokens)
            if i > 0:
                sps = SPS(in_dim, embed_dims[i], tau=tau, v_threshold=v_threshold)
                self.sps_modules.append(sps)
                in_dim = embed_dims[i]

            # Meta-SpikeFormer blocks
            stage = nn.ModuleList([
                MetaSpikeBlock(
                    dim=in_dim,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=1,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    tau=tau,
                    v_threshold=v_threshold,
                    use_groupnorm=use_groupnorm,
                )
                for _ in range(depths[i])
            ])
            self.stages.append(stage)

        # ---- Classification head ----
        self.head_norm = _norm(in_dim, use_groupnorm)
        self.head = nn.Linear(in_dim, num_classes)

        # ---- Init ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  [T, B, C, H, W]  or  [B, C, H, W]
                If [B, C, H, W], auto-repeat along dim 0 to [T, B, C, H, W].
        Returns:
            logits: [B, num_classes]  (averaged over T time-steps)
        """
        # Auto-handle missing T dimension
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        T, B, C, H, W = x.shape

        # ---- Patch embed ----
        x = x.reshape(T * B, C, H, W)
        x = self.patch_embed(x)
        x = self.patch_bn(x)
        x = x.reshape(T, B, self.embed_dims[0], H, W)
        x = self.patch_lif(x)                                     # [T,B,C,H,W] binary

        # Convert to token sequence
        H_cur, W_cur = H, W
        x = x.reshape(T, B, self.embed_dims[0], H_cur * W_cur)    # [T,B,C,N]
        x = x.permute(0, 1, 3, 2).contiguous()                    # [T,B,N,C]

        # ---- Stages ----
        sps_idx = 0
        for i, stage in enumerate(self.stages):
            if i > 0:
                x, H_cur, W_cur = self.sps_modules[sps_idx](x, H_cur, W_cur)
                sps_idx += 1
            for blk in stage:
                x = blk(x, H_cur, W_cur)

        # ---- Classification ----
        x = x.mean(dim=2)                                          # [T, B, C]  (pool tokens)
        x = x.mean(dim=0)                                          # [B, C]     (pool time)

        # Head
        if self.use_groupnorm:
            x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)        # GroupNorm needs extra dim
        else:
            x = self.head_norm(x)
        x = self.head(x)                                           # logits
        return x


# ---------------------------------------------------------------------------
#  Convenience builders
# ---------------------------------------------------------------------------

def meta_spikeformer_micro(**kwargs) -> MetaSpikeFormer:
    """Micro variant for quick CPU pipeline verification (~0.2M params)."""
    defaults = dict(
        img_size=32,
        in_channels=3,
        num_classes=100,
        embed_dims=(16, 32, 64, 128),
        depths=(1, 1, 1, 1),
        num_heads=(2, 4, 8, 16),
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        T=4,
        tau=2.0,
        v_threshold=1.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_groupnorm=True,
    )
    defaults.update(kwargs)
    return MetaSpikeFormer(**defaults)


def meta_spikeformer_tiny(**kwargs) -> MetaSpikeFormer:
    """Tiny variant for quick experiments (~1.7M params)."""
    defaults = dict(
        img_size=32,
        in_channels=3,
        num_classes=100,
        embed_dims=(32, 64, 128, 256),
        depths=(1, 2, 2, 1),
        num_heads=(2, 4, 8, 16),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        T=4,
        tau=2.0,
        v_threshold=1.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_groupnorm=True,
    )
    defaults.update(kwargs)
    return MetaSpikeFormer(**defaults)


def meta_spikeformer_cifar100(**kwargs) -> MetaSpikeFormer:
    """Default CIFAR-100 Meta-SpikeFormer (~11.6M params)."""
    defaults = dict(
        img_size=32,
        in_channels=3,
        num_classes=100,
        embed_dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        num_heads=(4, 8, 16, 32),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        T=4,
        tau=2.0,
        v_threshold=1.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_groupnorm=True,
    )
    defaults.update(kwargs)
    return MetaSpikeFormer(**defaults)
