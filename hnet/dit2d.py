"""
Bidirectional 2D Hierarchical Network Diffusion Transformer (H-Net-DiT, Design C)
with Rectified Flow (x-prediction parameterization) on 2D images.

Key architectural components:
  1. 2D Hilbert Curve Ordering: maps 2D micro-patch/pixel grid (H_g, W_g) <-> 1D sequence L = H_g * W_g
     so every contiguous 1D chunk [t_{k-1}+1 .. t_k] is a compact 2D spatial region.
  2. Outer Encoder & Decoder (`Bidirectional2DSSMBlock`, layout "m4"):
     - 2D Depthwise Conv2d (3x3) on the 2D grid for immediate 8-neighbor spatial mixing
     - Bidirectional Fused Triton EMA Scan (forward + backward along the 2D Hilbert curve)
     - AdaLN-Zero conditioning on (timestep t, class label y)
  3. Dynamic 2D Chunker (`RoutingModule2D` + `ChunkLayer2D`):
     - Predicts chunk boundaries via cosine dissimilarity along the 2D Hilbert curve
     - Aggregates chunk states (boundary + within-chunk mean pool) into M ~ L / N chunks
     - Computes exact 2D spatial centroids (x_k, y_k) and log-area for each dynamic chunk
  4. Main DiT (`DiTStack2D`, layout "T12"):
     - Bidirectional FlashAttention on the M compressed chunks
     - 2D Continuous Axial Partial RoPE (25% X-rotary, 25% Y-rotary, 50% content channels)
     - AdaLN-Zero + SwiGLU (3.6x D)
  5. Bidirectional Dechunker (`DeChunkLayer2D` + Sub-Chunk Relative 2D Offset + STE Residual):
     - Bidirectional confidence-weighted EMA scan across the M chunks
     - Broadcasts chunk k back to its constituent micro-patches + adds sub-chunk 2D offset (pos_t - c_k)
     - Combines with Encoder via Straight-Through Estimator (STE) + 0.5*I scaled-identity residual
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SwiGLU, rotate_half
from .dc import RoutingOutput, parallel_ema_scan, ratio_loss, ste_func
from .hilbert import build_hilbert_lut


@dataclass
class HNetDiT2DConfig:
    img_size: int = 64
    in_channels: int = 3
    patch_size: int = 2              # 2 -> 32x32 grid (L=1024); 1 -> 64x64 grid (L=4096)
    num_classes: int = 1000
    class_dropout_prob: float = 0.10

    # Hierarchy dimensions: [d_outer, d_main]
    d_model: List[int] = field(default_factory=lambda: [384, 768])
    d_cond: int = 384

    # Depth of outer encoder/decoder (2D-Conv + Bidirectional Hilbert SSM) and inner Main DiT
    num_encoder_blocks: int = 4
    num_decoder_blocks: int = 4
    num_main_blocks: int = 12

    head_dim: int = 64
    rotary_emb_dim: int = 32         # 50% of head_dim (16 for X, 16 for Y)
    mlp_expansion: float = 3.6

    # Dynamic chunking parameters
    target_compression: float = 4.0  # L=1024 -> M~256 chunks
    aux_loss_weight: float = 0.20
    routing_mode: str = "dynamic"    # "dynamic" (learned H-Net) or "uniform" (fixed 2x2 Hilbert blocks)

    # Per-stage learning rate multipliers
    lr_multipliers: List[float] = field(default_factory=lambda: [2.0, 1.0])

    # Rectified Flow x-prediction stability clamp on (1 - t) when converting x_1 -> v
    min_one_minus_t: float = 0.05

    @property
    def grid_size(self) -> int:
        assert self.img_size % self.patch_size == 0
        return self.img_size // self.patch_size

    @property
    def seq_len(self) -> int:
        g = self.grid_size
        return g * g

    @classmethod
    def dit2d_small(cls, patch_size: int = 2, routing_mode: str = "dynamic") -> "HNetDiT2DConfig":
        """~45M parameter 2D H-Net-DiT for fast experimentation."""
        return cls(
            patch_size=patch_size,
            d_model=[256, 512],
            d_cond=256,
            num_encoder_blocks=4,
            num_decoder_blocks=4,
            num_main_blocks=8,
            head_dim=64,
            rotary_emb_dim=32,
            target_compression=4.0,
            routing_mode=routing_mode,
        )

    @classmethod
    def dit2d_base(cls, patch_size: int = 2, routing_mode: str = "dynamic") -> "HNetDiT2DConfig":
        """~130M parameter 2D H-Net-DiT (m4 outer d=384, T12 main d=768)."""
        return cls(
            patch_size=patch_size,
            d_model=[384, 768],
            d_cond=384,
            num_encoder_blocks=4,
            num_decoder_blocks=4,
            num_main_blocks=12,
            head_dim=64,
            rotary_emb_dim=32,
            target_compression=4.0,
            routing_mode=routing_mode,
        )

    @classmethod
    def dit2d_large(cls, patch_size: int = 2, routing_mode: str = "dynamic") -> "HNetDiT2DConfig":
        """~240M parameter 2D H-Net-DiT (m4 outer d=512, T16 main d=896)."""
        return cls(
            patch_size=patch_size,
            d_model=[512, 896],
            d_cond=512,
            num_encoder_blocks=4,
            num_decoder_blocks=4,
            num_main_blocks=16,
            head_dim=64,
            rotary_emb_dim=32,
            target_compression=4.0,
            routing_mode=routing_mode,
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepClassEmbedder(nn.Module):
    """
    Embeds continuous flow timestep t in [0, 1] + discrete class label y in [0 .. num_classes]
    (where index `num_classes` is the null/unconditional token for CFG).
    """

    def __init__(self, d_cond: int, num_classes: int, freq_dim: int = 256):
        super().__init__()
        self.d_cond = d_cond
        self.num_classes = num_classes
        self.freq_dim = freq_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(freq_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        # +1 for unconditional / CFG null class token
        self.class_embed = nn.Embedding(num_classes + 1, d_cond)
        nn.init.normal_(self.class_embed.weight, std=0.02)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = (t.float() * 1000.0).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self._sinusoidal_embedding(t).to(self.time_mlp[0].weight.dtype)
        c = self.time_mlp(t_emb)
        if y is None:
            y = torch.full((t.shape[0],), self.num_classes, device=t.device, dtype=torch.long)
        c = c + self.class_embed(y)
        return F.silu(c)


class ContinuousAxialRotary2D(nn.Module):
    """
    2D Continuous Axial Partial RoPE for dynamic chunks with arbitrary continuous 2D centroids (x_k, y_k).
    Rotates:
      - channels [0 .. rotary_dim//2) by continuous coordinate x_k
      - channels [rotary_dim//2 .. rotary_dim) by continuous coordinate y_k
      - channels [rotary_dim .. head_dim) remain unrotated position-invariant content channels.
    """

    def __init__(self, rotary_dim: int, base: float = 100.0):
        super().__init__()
        assert rotary_dim % 4 == 0, f"2D rotary_dim ({rotary_dim}) must be divisible by 4"
        self.rotary_dim = rotary_dim
        self.axis_dim = rotary_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, centroids_grid_xy: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            centroids_grid_xy: (B, M, 2) float32 continuous grid coordinates in [0, grid_size)
        Returns:
            cos, sin: (B, 1, M, rotary_dim)
        """
        cx = centroids_grid_xy[..., 0].float()  # (B, M)
        cy = centroids_grid_xy[..., 1].float()  # (B, M)

        freqs_x = torch.einsum("bm,d->bmd", cx, self.inv_freq)  # (B, M, axis_dim // 2)
        freqs_y = torch.einsum("bm,d->bmd", cy, self.inv_freq)  # (B, M, axis_dim // 2)

        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)           # (B, M, axis_dim)
        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)           # (B, M, axis_dim)
        emb = torch.cat([emb_x, emb_y], dim=-1).to(dtype)       # (B, M, rotary_dim)

        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


class Bidirectional2DSSMBlock(nn.Module):
    """
    Outer Encoder/Decoder block operating at full micro-patch resolution L = H_g * W_g:
      1. AdaLN-modulated RMSNorm
      2. 2D Depthwise Conv2d (3x3) on the unflattened 2D grid (immediate 8-neighbor spatial receptive field)
      3. Bidirectional Fused Triton EMA Scan (forward + backward along the 2D Hilbert curve)
      4. Gated output projection
    """

    def __init__(self, d_model: int, d_cond: int, grid_size: int):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.adaLN = nn.Linear(d_cond, 3 * d_model, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.adaLN.weight._no_reinit = True

        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=True,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        hilbert_to_raster: torch.Tensor,
        raster_to_hilbert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) in 2D Hilbert order
            cond: (B, D_cond) timestep + class conditioning
            hilbert_to_raster: (L,) index mapping
            raster_to_hilbert: (L,) inverse index mapping
        """
        B, L, D = x.shape
        G = self.grid_size

        shift, scale, gate_res = self.adaLN(cond).chunk(3, dim=-1)
        h = modulate(self.norm1(x), shift, scale)

        u, gate, forget_f, forget_b = self.in_proj(h).chunk(4, dim=-1)

        # 1. 2D Depthwise Conv2d on the true 2D spatial grid
        u_raster = u.index_select(1, raster_to_hilbert)  # (B, G*G, D)
        u_2d = u_raster.transpose(1, 2).reshape(B, D, G, G)
        u_conv2d = F.silu(self.conv2d(u_2d))
        u_hilbert = u_conv2d.reshape(B, D, L).transpose(1, 2).index_select(1, hilbert_to_raster)

        # 2. Bidirectional Fused Triton EMA Scan along the 2D Hilbert curve
        pf = torch.sigmoid(forget_f.float()).mean(dim=-1, keepdim=True)
        pb = torch.sigmoid(forget_b.float()).mean(dim=-1, keepdim=True)

        rec_fwd = parallel_ema_scan(u_hilbert, pf).to(x.dtype)
        rec_bwd = parallel_ema_scan(u_hilbert.flip(1), pb.flip(1)).flip(1).to(x.dtype)

        rec_out = 0.5 * (rec_fwd + rec_bwd) * F.silu(gate)
        out = self.out_proj(rec_out)
        return x + (1.0 + gate_res.unsqueeze(1)) * out


class RoutingModule2D(nn.Module):
    """
    Predicts dynamic 2D chunk boundaries along the 2D Hilbert curve via cosine dissimilarity
    between adjacent states (t-1, t).
    Position t = L - 1 is always closed (`boundary_mask[:, -1] = True`) so every micro-patch
    in the 2D image belongs to a complete chunk k in [0 .. M-1].
    Supports `routing_mode="uniform"` for apples-to-apples static 2x2 block comparison.
    """

    def __init__(self, d_model: int, target_compression: float = 4.0, routing_mode: str = "dynamic"):
        super().__init__()
        self.d_model = d_model
        self.target_compression = target_compression
        self.routing_mode = routing_mode

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.bos_embed = nn.Parameter(torch.empty(d_model))
        with torch.no_grad():
            nn.init.orthogonal_(self.q_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight)
            nn.init.normal_(self.bos_embed, mean=0.0, std=0.02)
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True
        self.bos_embed._no_reinit = True

    def forward(self, hidden_states: torch.Tensor) -> RoutingOutput:
        B, L, D = hidden_states.shape
        device = hidden_states.device

        if self.routing_mode == "uniform":
            stride = max(int(round(self.target_compression)), 1)
            idx = torch.arange(L, device=device)
            b_mask = ((idx + 1) % stride == 0) | (idx == L - 1)
            boundary_mask = b_mask.unsqueeze(0).expand(B, L).clone()
            p_val = 1.0 / self.target_compression
            p_boundary = torch.full((B, L), p_val, device=device, dtype=hidden_states.dtype)
            boundary_prob = torch.stack([1.0 - p_boundary, p_boundary], dim=-1)
            selected_probs = torch.ones((B, L, 1), device=device, dtype=hidden_states.dtype)
            return RoutingOutput(
                boundary_prob=boundary_prob,
                boundary_mask=boundary_mask,
                selected_probs=selected_probs,
            )

        bos = self.bos_embed.to(dtype=hidden_states.dtype).view(1, 1, D).expand(B, 1, D)
        prev_states = torch.cat([bos, hidden_states[:, :-1]], dim=1)

        q = F.normalize(self.q_proj(prev_states), dim=-1)
        k = F.normalize(self.k_proj(hidden_states), dim=-1)

        cos_sim = torch.einsum("bld,bld->bl", q, k)
        p_boundary = torch.clamp((1.0 - cos_sim) / 2.0, min=0.0, max=1.0)

        boundary_prob = torch.stack([1.0 - p_boundary, p_boundary], dim=-1)  # (B, L, 2)
        selected_idx = torch.argmax(boundary_prob, dim=-1)                   # (B, L)
        boundary_mask = selected_idx == 1

        # Always close the final chunk at t = L - 1 so all L positions belong to a valid chunk
        boundary_mask = boundary_mask.clone()
        boundary_mask[:, -1] = True

        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))

        return RoutingOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )


@dataclass
class Chunk2DOutput:
    chunk_states: torch.Tensor       # (B, M, D)
    next_mask: torch.Tensor          # (B, M) bool mask of valid chunks
    sorted_indices: torch.Tensor     # (B, M) Hilbert index t_k where chunk k ends
    chunk_idx: torch.Tensor          # (B, L) chunk assignment k in [0 .. M-1] for each Hilbert position t
    centroids_norm_xy: torch.Tensor  # (B, M, 2) normalized centroids in [-1, 1]^2
    centroids_grid_xy: torch.Tensor  # (B, M, 2) grid centroids in [0, grid_size)^2
    chunk_sizes: torch.Tensor        # (B, M, 1) number of micro-patches in each chunk


class ChunkLayer2D(nn.Module):
    """
    Packs variable-size 2D Hilbert chunks from (B, L, D) -> (B, M, D):
      - Assigns every position t in [0 .. L-1] to chunk k = cumsum(b)_t - b_t in [0 .. M-1]
      - Combines boundary state h[t_k] with within-chunk mean pool via scatter_add_
      - Computes exact 2D spatial centroids (x_k, y_k) and chunk sizes s_k
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        pos_xy_norm: torch.Tensor,
        pos_xy_grid: torch.Tensor,
    ) -> Chunk2DOutput:
        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        num_chunks = boundary_mask.sum(dim=-1)          # (B,)
        max_chunks = max(int(num_chunks.max().item()), 1)

        # 1. Gather boundary end positions t_k for each chunk k
        token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        sorted_indices = torch.argsort(token_idx, dim=1)[:, :max_chunks]  # (B, M)

        boundary_states = torch.gather(
            hidden_states,
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(B, max_chunks, D),
        )
        next_mask = torch.arange(max_chunks, device=device)[None, :] < num_chunks[:, None]

        # 2. Assign each position t to chunk k = cumsum(b)_t - b_t in [0 .. M-1]
        b_long = boundary_mask.long()
        chunk_idx = torch.clamp(torch.cumsum(b_long, dim=1) - b_long, min=0, max=max_chunks - 1)  # (B, L)

        # 3. Within-chunk mean pooling + 2D spatial centroid computation via scatter_add_
        ones = torch.ones((B, L, 1), device=device, dtype=torch.float32)
        chunk_counts = torch.zeros((B, max_chunks, 1), device=device, dtype=torch.float32)
        chunk_counts.scatter_add_(1, chunk_idx.unsqueeze(-1), ones)
        chunk_counts_safe = chunk_counts.clamp(min=1.0)

        sum_states = torch.zeros((B, max_chunks, D), device=device, dtype=torch.float32)
        sum_states.scatter_add_(1, chunk_idx.unsqueeze(-1).expand(B, L, D), hidden_states.float())
        mean_states = (sum_states / chunk_counts_safe).to(dtype)

        # Combine boundary representation + chunk mean representation
        chunk_states = 0.5 * (boundary_states + mean_states)

        # Compute exact 2D centroids in normalized [-1, 1]^2 and grid [0, grid_size)^2 coordinates
        pos_norm_exp = pos_xy_norm.unsqueeze(0).expand(B, L, 2)
        pos_grid_exp = pos_xy_grid.unsqueeze(0).expand(B, L, 2)
        idx_2d = chunk_idx.unsqueeze(-1).expand(B, L, 2)

        sum_norm_xy = torch.zeros((B, max_chunks, 2), device=device, dtype=torch.float32)
        sum_grid_xy = torch.zeros((B, max_chunks, 2), device=device, dtype=torch.float32)
        sum_norm_xy.scatter_add_(1, idx_2d, pos_norm_exp)
        sum_grid_xy.scatter_add_(1, idx_2d, pos_grid_exp)

        centroids_norm_xy = sum_norm_xy / chunk_counts_safe
        centroids_grid_xy = sum_grid_xy / chunk_counts_safe

        return Chunk2DOutput(
            chunk_states=chunk_states,
            next_mask=next_mask,
            sorted_indices=sorted_indices,
            chunk_idx=chunk_idx,
            centroids_norm_xy=centroids_norm_xy,
            centroids_grid_xy=centroids_grid_xy,
            chunk_sizes=chunk_counts,
        )


class DeChunkLayer2D(nn.Module):
    """
    Bidirectional 2D DeChunker:
      1. Smooths chunk states (B, M, D) via forward + backward confidence-weighted Triton EMA scan
      2. Broadcasts chunk k back to all positions t belonging to chunk k (`chunk_idx`)
      3. Adds a learned Sub-Chunk Relative 2D Offset embedding `OffsetMLP(pos_xy[t] - centroid_xy[k])`
         so the Decoder knows exact sub-chunk pixel coordinates inside merged background regions.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.offset_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        # Zero-init final offset projection so initial dechunking starts purely from chunk_states
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)
        self.offset_mlp[-1].weight._no_reinit = True

    def forward(
        self,
        chunk_states: torch.Tensor,
        boundary_prob: torch.Tensor,
        chunk_out: Chunk2DOutput,
        pos_xy_norm: torch.Tensor,
    ) -> torch.Tensor:
        B, M, D = chunk_states.shape
        L = chunk_out.chunk_idx.shape[1]
        orig_dtype = chunk_states.dtype

        # 1. Gather boundary confidence p_k for each chunk k
        p_all = torch.clamp(boundary_prob[..., 1].float(), min=1e-4, max=1.0 - 1e-4)
        p_chunk = torch.gather(p_all, dim=1, index=chunk_out.sorted_indices).unsqueeze(-1)  # (B, M, 1)

        # Mask out invalid padded chunk positions (k >= num_chunks[b]) so they don't corrupt backward scan
        valid_f = chunk_out.next_mask.unsqueeze(-1).float()
        p_masked = p_chunk * valid_f
        z_masked = chunk_states.float() * valid_f

        ema_fwd = parallel_ema_scan(z_masked, p_masked)
        ema_bwd = parallel_ema_scan(z_masked.flip(1), p_masked.flip(1)).flip(1)
        smoothed = (0.5 * (ema_fwd + ema_bwd)).to(orig_dtype)

        # 2. Broadcast chunk k back to all positions t in chunk k
        dechunked = torch.gather(
            smoothed,
            dim=1,
            index=chunk_out.chunk_idx.unsqueeze(-1).expand(B, L, D),
        )

        # 3. Sub-chunk relative 2D coordinate offset: delta_xy = pos_xy[t] - centroid_xy[chunk_idx[t]]
        chunk_centroids_per_t = torch.gather(
            chunk_out.centroids_norm_xy,
            dim=1,
            index=chunk_out.chunk_idx.unsqueeze(-1).expand(B, L, 2),
        )
        delta_xy = (pos_xy_norm.unsqueeze(0) - chunk_centroids_per_t).to(orig_dtype)
        dechunked = dechunked + self.offset_mlp(delta_xy)

        return dechunked


class DiTBlock2D(nn.Module):
    """
    Bidirectional DiT Block operating on M dynamic chunks with:
      - AdaLN-Zero conditioning on (timestep t, class y)
      - 2D Continuous Axial Partial RoPE using chunk centroids (x_k, y_k)
      - Bidirectional FlashAttention SDPA with padding mask `next_mask`
      - SwiGLU MLP (3.6x D)
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        head_dim: int = 64,
        rotary_emb_dim: int = 32,
        mlp_expansion: float = 3.6,
    ):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = d_model // head_dim
        self.rotary_emb_dim = rotary_emb_dim

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, expansion=mlp_expansion)

        self.adaLN = nn.Linear(d_cond, 6 * d_model, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.adaLN.weight._no_reinit = True

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, M, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(6, dim=-1)

        # 1. Bidirectional Multi-Head Attention with 2D Continuous Partial RoPE
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(h).view(B, M, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, M, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary_emb_dim > 0:
            r_dim = self.rotary_emb_dim
            q_rot, q_pass = q[..., :r_dim], q[..., r_dim:]
            k_rot, k_pass = k[..., :r_dim], k[..., r_dim:]
            q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, D)
        x = x + (1.0 + gate_msa.unsqueeze(1)) * self.out_proj(attn_out)

        # 2. SwiGLU FFN
        h2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + (1.0 + gate_mlp.unsqueeze(1)) * self.mlp(h2)
        return x


class HNetDiT2D(nn.Module):
    """
    Complete Bidirectional 2D H-Net-DiT for Rectified Flow (x-prediction parameterization).
    """

    def __init__(self, config: HNetDiT2DConfig):
        super().__init__()
        self.config = config
        d_outer, d_main = config.d_model[0], config.d_model[1]
        self.d_outer = d_outer
        self.d_main = d_main
        self.patch_dim = config.in_channels * config.patch_size * config.patch_size

        # 1. Precompute 2D Hilbert curve lookup tables for (grid_size, grid_size)
        h2r, r2h, pos_norm, pos_grid = build_hilbert_lut(config.grid_size)
        self.register_buffer("hilbert_to_raster", h2r, persistent=False)
        self.register_buffer("raster_to_hilbert", r2h, persistent=False)
        self.register_buffer("pos_xy_norm", pos_norm, persistent=False)
        self.register_buffer("pos_xy_grid", pos_grid, persistent=False)

        # 2. Conditioning embedder (t, y)
        self.cond_embedder = TimestepClassEmbedder(config.d_cond, config.num_classes)

        # 3. Micro-patch stem + 2D coordinate embedding
        self.patch_embed = nn.Linear(self.patch_dim, d_outer)
        self.pos_embed_outer = nn.Linear(2, d_outer)

        # 4. Outer Stage 0 Encoder (Bidirectional 2D Conv + Hilbert SSM)
        self.encoder = nn.ModuleList(
            [
                Bidirectional2DSSMBlock(d_outer, config.d_cond, config.grid_size)
                for _ in range(config.num_encoder_blocks)
            ]
        )

        # 5. Dynamic 2D Chunker
        self.routing_module = RoutingModule2D(
            d_outer,
            target_compression=config.target_compression,
            routing_mode=config.routing_mode,
        )
        self.chunk_layer = ChunkLayer2D()

        # Dimension expansion d_outer -> d_main (learned zero-init pad matching H-Net) + chunk log-size embed
        if d_main > d_outer:
            self.pad_dimension = nn.Parameter(torch.zeros(d_main - d_outer))
        else:
            self.pad_dimension = None
        self.chunk_size_embed = nn.Sequential(
            nn.Linear(1, d_main // 4),
            nn.SiLU(),
            nn.Linear(d_main // 4, d_main),
        )

        # 6. Inner Main DiT (T12) with 2D Continuous Axial Partial RoPE
        self.rope_2d = ContinuousAxialRotary2D(
            rotary_dim=config.rotary_emb_dim,
            base=float(max(config.grid_size * 2, 64)),
        )
        self.main_blocks = nn.ModuleList(
            [
                DiTBlock2D(
                    d_model=d_main,
                    d_cond=config.d_cond,
                    head_dim=config.head_dim,
                    rotary_emb_dim=config.rotary_emb_dim,
                    mlp_expansion=config.mlp_expansion,
                )
                for _ in range(config.num_main_blocks)
            ]
        )
        self.main_norm = nn.RMSNorm(d_main, eps=1e-5)

        # 7. Bidirectional DeChunker + 0.5*I Scaled-Identity Residual
        self.dechunk_layer = DeChunkLayer2D(d_outer)
        self.residual_proj = nn.Linear(d_outer, d_outer, dtype=torch.float32)
        with torch.no_grad():
            nn.init.eye_(self.residual_proj.weight)
            self.residual_proj.weight.mul_(0.5)
            nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True

        # 8. Outer Stage 0 Decoder (Bidirectional 2D Conv + Hilbert SSM)
        self.decoder = nn.ModuleList(
            [
                Bidirectional2DSSMBlock(d_outer, config.d_cond, config.grid_size)
                for _ in range(config.num_decoder_blocks)
            ]
        )

        # 9. Final AdaLN + Linear projection to clean micro-patch x_1 residual over x_t
        self.final_norm = nn.RMSNorm(d_outer, eps=1e-5)
        self.final_adaLN = nn.Linear(config.d_cond, 2 * d_outer, bias=True)
        self.final_head = nn.Linear(d_outer, self.patch_dim, bias=True)

        self._init_weights()
        self._tag_stage_params()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if getattr(m.weight, "_no_reinit", False):
                    continue
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Depth-scaled residual init for out_proj and fc2
        total_residuals = (
            self.config.num_encoder_blocks
            + 2 * self.config.num_main_blocks
            + self.config.num_decoder_blocks
        )
        res_std = 0.02 / math.sqrt(max(total_residuals, 1))
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=res_std)

        # Zero-init final AdaLN so initial output starts near x_t
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

    def _tag_stage_params(self):
        for p in self.parameters():
            p._stage_idx = 0
        for p in self.main_blocks.parameters():
            p._stage_idx = 1
        for p in self.main_norm.parameters():
            p._stage_idx = 1
        for p in self.chunk_size_embed.parameters():
            p._stage_idx = 1

    def patchify_hilbert(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, C, H, W) image -> (B, L, patch_dim) ordered along the 2D Hilbert curve.
        """
        B, C, H, W = img.shape
        p = self.config.patch_size
        G = self.config.grid_size
        # (B, C, G, p, G, p) -> (B, G, G, C, p, p) -> (B, G*G, C*p*p)
        patches_raster = (
            img.view(B, C, G, p, G, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, G * G, self.patch_dim)
        )
        return patches_raster.index_select(1, self.hilbert_to_raster)

    def unpatchify_hilbert(self, patches_hilbert: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, L, patch_dim) in 2D Hilbert order -> (B, C, H, W) image.
        """
        B, L, _ = patches_hilbert.shape
        C = self.config.in_channels
        p = self.config.patch_size
        G = self.config.grid_size
        patches_raster = patches_hilbert.index_select(1, self.raster_to_hilbert)
        img = (
            patches_raster.view(B, G, G, C, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C, G * p, G * p)
        )
        return img

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, RoutingOutput, Chunk2DOutput]:
        """
        Predict clean image x_1 from noisy interpolant x_t = t * x_1 + (1 - t) * x_0.

        Args:
            x_t: (B, C, H, W) noisy image at flow time t in [0, 1]
            t: (B,) flow time in [0, 1]
            y: (B,) class labels in [0 .. num_classes - 1] (or num_classes for null token)
        Returns:
            x1_pred: (B, C, H, W) predicted clean image
            router_out: RoutingOutput with boundary_prob and boundary_mask
            chunk_out: Chunk2DOutput with 2D centroids and chunk assignments
        """
        B = x_t.shape[0]
        cond = self.cond_embedder(t, y)  # (B, D_cond)

        # 1. Patchify in 2D Hilbert order + add 2D coordinate embedding
        patches_in = self.patchify_hilbert(x_t)  # (B, L, patch_dim)
        h = self.patch_embed(patches_in) + self.pos_embed_outer(self.pos_xy_norm.to(x_t.dtype)).unsqueeze(0)

        # 2. Outer Encoder (m4: 2D Conv + Bidirectional Hilbert SSM)
        for blk in self.encoder:
            h = blk(h, cond, self.hilbert_to_raster, self.raster_to_hilbert)
        h_enc = h

        # 3. Dynamic 2D Routing & Chunking: (B, L, D_outer) -> (B, M, D_outer)
        router_out = self.routing_module(h_enc)
        chunk_out = self.chunk_layer(
            h_enc, router_out.boundary_mask, self.pos_xy_norm, self.pos_xy_grid
        )
        z = chunk_out.chunk_states  # (B, M, D_outer)
        M = z.shape[1]

        # 4. Expand to Main DiT dimension + add chunk log-size embedding
        if self.pad_dimension is not None:
            pad = self.pad_dimension.to(dtype=z.dtype).view(1, 1, -1).expand(B, M, -1)
            z = torch.cat([z, pad], dim=-1)

        log_size = torch.log(chunk_out.chunk_sizes.clamp(min=1.0) / self.config.target_compression).to(z.dtype)
        z = z + self.chunk_size_embed(log_size)

        # 5. Main DiT (T12) with 2D Continuous Axial Partial RoPE on chunk centroids
        cos, sin = self.rope_2d(chunk_out.centroids_grid_xy, dtype=z.dtype)
        # Key padding mask: (B, 1, 1, M) bool mask where True = allowed to attend
        attn_mask = None
        if not chunk_out.next_mask.all():
            attn_mask = chunk_out.next_mask[:, None, None, :]

        for blk in self.main_blocks:
            z = blk(z, cond, cos, sin, attn_mask=attn_mask)
        z = self.main_norm(z)[..., : self.d_outer]

        # 6. Bidirectional DeChunking + Sub-Chunk Relative 2D Offset + STE Residual
        z_dechunk = self.dechunk_layer(
            z, router_out.boundary_prob, chunk_out, self.pos_xy_norm
        )
        residual = self.residual_proj(h_enc.float()).to(z_dechunk.dtype)
        h_dec = z_dechunk * ste_func(router_out.selected_probs.to(z_dechunk.dtype)) + residual

        # 7. Outer Decoder (m4: 2D Conv + Bidirectional Hilbert SSM)
        for blk in self.decoder:
            h_dec = blk(h_dec, cond, self.hilbert_to_raster, self.raster_to_hilbert)

        # 8. Final AdaLN + clean image x_1 prediction
        shift_f, scale_f = self.final_adaLN(cond).chunk(2, dim=-1)
        h_final = modulate(self.final_norm(h_dec), shift_f, scale_f)
        patches_out = self.final_head(h_final)
        x1_pred = self.unpatchify_hilbert(patches_out)

        return x1_pred, router_out, chunk_out

    def forward_loss(
        self,
        x_1: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Rectified Flow training step with x-prediction parameterization (JiT style):
          x_t = t * x_1 + (1 - t) * x_0,   x_0 ~ N(0, I)
          v_target = x_1 - x_0
          v_pred = (x1_pred - x_t) / clamp(1 - t, min=min_one_minus_t)
          flow_loss = MSE(v_pred, v_target)
        """
        B = x_1.shape[0]
        device = x_1.device

        # Logit-normal timestep sampling (standard for Rectified Flow / SD3 / JiT)
        if t is None:
            u = torch.randn(B, device=device, dtype=torch.float32)
            t = torch.sigmoid(u).clamp(min=1e-4, max=1.0 - 1e-4)

        if noise is None:
            noise = torch.randn_like(x_1)

        # Classifier-free guidance label dropout during training
        if y is not None and self.training and self.config.class_dropout_prob > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.class_dropout_prob
            y_in = torch.where(drop_mask, torch.full_like(y, self.config.num_classes), y)
        else:
            y_in = y

        t_view = t.view(B, 1, 1, 1).to(x_1.dtype)
        x_t = t_view * x_1 + (1.0 - t_view) * noise
        v_target = x_1.float() - noise.float()

        x1_pred, router_out, _ = self.forward(x_t, t, y_in)

        # Convert x1_pred to velocity v_pred = (x1_pred - x_t) / (2 * sqrt((1 - t)^2 + 0.1))
        # Note: (x1_pred - x_t) - (1 - t)*(x_1 - x_0) == (x1_pred - x_1), so damped v-loss
        # weights ||x1_pred - x_1||^2 by 1 / (4 * ((1 - t)^2 + 0.1)) (mean weight ~ 1.0 over t in [0, 1]),
        # preventing 400x gradient spikes near t -> 1 while keeping exact x-prediction velocity semantics.
        denom = (2.0 * torch.sqrt((1.0 - t) ** 2 + 0.1)).view(B, 1, 1, 1)
        v_pred = (x1_pred.float() - x_t.float()) / denom
        v_target_damped = ((1.0 - t).view(B, 1, 1, 1) / denom) * v_target

        flow_loss = F.mse_loss(v_pred, v_target_damped)
        x1_mse = F.mse_loss(x1_pred.float(), x_1.float())

        # Auxiliary Switch-Transformer ratio loss for dynamic routing
        valid_mask = torch.ones((B, self.config.seq_len), device=device, dtype=torch.bool)
        r_loss = ratio_loss(router_out, valid_mask, self.config.target_compression)

        if self.config.routing_mode == "dynamic":
            total_loss = flow_loss + self.config.aux_loss_weight * r_loss
        else:
            total_loss = flow_loss

        b_frac = router_out.boundary_mask.float().mean().item()
        comp_ratio = 1.0 / max(b_frac, 1e-6)
        avg_prob = router_out.boundary_prob[..., 1].float().mean().item()

        metrics = {
            "loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "x1_mse": x1_mse.item(),
            "ratio_loss": r_loss.item(),
            "compression_ratio": comp_ratio,
            "boundary_prob": avg_prob,
        }
        return total_loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        steps: int = 50,
        cfg_scale: float = 2.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        return_trajectories: bool = False,
    ):
        """
        Generate images via Rectified Flow ODE integration from t=0 (noise) -> t=1 (clean image)
        using x-prediction + Classifier-Free Guidance (CFG).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(
                (batch_size, self.config.in_channels, self.config.img_size, self.config.img_size),
                device=device,
                generator=gen,
            )
        else:
            x = torch.randn(
                (batch_size, self.config.in_channels, self.config.img_size, self.config.img_size),
                device=device,
            )

        if y is None:
            y = torch.randint(0, self.config.num_classes, (batch_size,), device=device)
        y_null = torch.full_like(y, self.config.num_classes)

        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
        snapshots = []

        for i in range(steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full((batch_size,), t_cur.item(), device=device, dtype=torch.float32)

            if cfg_scale > 1.0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
                y_in = torch.cat([y, y_null], dim=0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    x1_both, router_both, chunk_both = self.forward(x_in, t_in, y_in)
                x1_cond, x1_uncond = x1_both.float().chunk(2, dim=0)
                x1_pred = x1_uncond + cfg_scale * (x1_cond - x1_uncond)
                b_mask = router_both.boundary_mask[:batch_size]
                c_idx = chunk_both.chunk_idx[:batch_size]
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    x1_pred, router_out, chunk_out = self.forward(x, t_batch, y)
                x1_pred = x1_pred.float()
                b_mask = router_out.boundary_mask
                c_idx = chunk_out.chunk_idx

            # Soft clamp x1_pred for high-CFG stability
            x1_pred = x1_pred.clamp(-1.0, 1.0) if cfg_scale > 1.0 else x1_pred.clamp(-1.5, 1.5)

            # Exact Euler step along the linear interpolant toward x1_pred:
            # x_{t_{i+1}} = x_{t_i} + ((t_{i+1} - t_i) / (1 - t_i)) * (x1_pred - x_{t_i})
            alpha = (dt / (1.0 - t_cur).clamp(min=1e-5)).clamp(min=0.0, max=1.0)
            x = x + alpha * (x1_pred - x)

            if return_trajectories and (
                i in (0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1)
            ):
                b_2d, c_2d = self.hilbert_mask_to_2d(b_mask, c_idx)
                snapshots.append(
                    {
                        "step": i,
                        "t": t_cur.item(),
                        "x_t": x.detach().cpu(),
                        "x1_pred": x1_pred.detach().cpu(),
                        "boundary_2d": b_2d.detach().cpu(),
                        "chunk_id_2d": c_2d.detach().cpu(),
                        "compression": 1.0 / max(b_mask.float().mean().item(), 1e-6),
                    }
                )

        x_final = x.clamp(-1.0, 1.0)
        if return_trajectories:
            return x_final, snapshots
        return x_final

    def hilbert_mask_to_2d(
        self, boundary_mask: torch.Tensor, chunk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map 1D Hilbert boundary_mask (B, L) and chunk_idx (B, L) back to 2D grid (B, G, G).
        """
        B, L = boundary_mask.shape
        G = self.config.grid_size
        b_raster = boundary_mask.index_select(1, self.raster_to_hilbert).view(B, G, G)
        c_raster = chunk_idx.index_select(1, self.raster_to_hilbert).view(B, G, G)
        return b_raster, c_raster
