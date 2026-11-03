"""
Backbone sequence blocks for H-Net matching `goombalab/hnet` and Cartesia Part 1:
  - Partial RoPE (`rotary_emb_dim = head_dim // 2`, rotating only the first 50% of head channels)
  - SwiGLU MLP with intermediate dim = round_to_128(3.6 * d_model)
  - Layout string parser supporting:
      'm' = SSM mixer only (no FFN, used in outer 'm4' Encoder/Decoder)
      'M' = SSM mixer + SwiGLU FFN
      't' = Causal Attention only (no FFN)
      'T' = Causal Attention + SwiGLU FFN (used in 'T12'/'T16' Main Network and 'T1m4' hybrid stage)
  - Sliding-window causal attention when `window_size > 0`
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dc import parallel_ema_scan


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Partial Rotary Position Embedding (RoPE).
    Applies rotary embeddings to the first `rotary_dim` channels of each attention head
    (in `goombalab/hnet`, `rotary_dim = head_dim // 2`).
    """

    def __init__(self, rotary_dim: int, base: float = 10000.0):
        super().__init__()
        assert rotary_dim % 2 == 0, f"rotary_dim ({rotary_dim}) must be even"
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0):
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class SwiGLU(nn.Module):
    """
    SwiGLU FFN with intermediate dimension set to `3.6 * d_model` rounded to nearest multiple of 128
    (matching Cartesia's Part 1 scaling ladder specification).
    """

    def __init__(self, d_model: int, expansion: float = 3.6, multiple_of: int = 128):
        super().__init__()
        hidden = int(round((expansion * d_model) / multiple_of)) * multiple_of
        hidden = max(hidden, multiple_of)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)  # named fc2 for 1/sqrt(n_residuals) init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.w1(x)) * self.w3(x))


@dataclass
class SSMLayerState:
    conv_buf: torch.Tensor   # (B, D_inner, K-1)
    ssm_state: torch.Tensor  # (B, D_inner)


@dataclass
class KVCacheState:
    k: torch.Tensor          # (B, H, L_cur, Dh)
    v: torch.Tensor          # (B, H, L_cur, Dh)


class CausalSSMBlock(nn.Module):
    """
    Mamba-inspired State Space / Gated Recurrence Block for outer Encoder & Decoder stages.
    - When `has_mlp=False` ('m' in arch_layout, e.g. 'm4'): pure SSM mixer block without FFN.
    - When `has_mlp=True` ('M' in arch_layout): SSM mixer + SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        has_mlp: bool = False,
        mlp_expansion: float = 3.6,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.has_mlp = has_mlp

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=True,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if has_mlp:
            self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
            self.mlp = SwiGLU(d_model, expansion=mlp_expansion)
        else:
            self.norm2 = None
            self.mlp = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> SSMLayerState:
        return SSMLayerState(
            conv_buf=torch.zeros(batch_size, self.d_model, self.kernel_size - 1, device=device, dtype=dtype),
            ssm_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        u, gate, forget_logits = self.in_proj(h).chunk(3, dim=-1)

        # 1. Causal depthwise conv1d along sequence dimension
        u_conv = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # 2. Data-dependent gated linear recurrence: s_t = p_t * u_conv_t + (1 - p_t) * s_{t-1}
        p_forget = torch.sigmoid(forget_logits.float()).mean(dim=-1, keepdim=True)
        if mask is not None:
            p_forget = p_forget * mask.unsqueeze(-1).float()
        rec_out = parallel_ema_scan(u_conv, p_forget).to(x.dtype)

        # 3. Output gating & residual
        x = x + self.out_proj(rec_out * F.silu(gate))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x: torch.Tensor, state: SSMLayerState) -> torch.Tensor:
        """O(1) single-step update for x of shape (B, 1, D)."""
        h = self.norm1(x)
        u, gate, forget_logits = self.in_proj(h).chunk(3, dim=-1)

        u_col = u.transpose(1, 2)
        conv_in = torch.cat([state.conv_buf, u_col], dim=-1)
        state.conv_buf.copy_(conv_in[:, :, 1:])
        weight = self.conv1d.weight.squeeze(1)
        u_conv = (conv_in * weight.unsqueeze(0)).sum(dim=-1) + self.conv1d.bias
        u_conv = F.silu(u_conv)

        p_forget = torch.sigmoid(forget_logits.squeeze(1).float()).mean(dim=-1, keepdim=True).to(x.dtype)
        new_ssm = p_forget * u_conv + (1.0 - p_forget) * state.ssm_state
        state.ssm_state.copy_(new_ssm)

        x = x + self.out_proj(new_ssm.unsqueeze(1) * F.silu(gate))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class CausalTransformerBlock(nn.Module):
    """
    Causal Transformer block with:
      - Partial RoPE (`rotary_emb_dim` channels rotated per head, e.g. 50% of `head_dim`)
      - Global Causal FlashAttention (`window_size == -1`) or Sliding-Window Causal Attention (`window_size > 0`)
      - Optional SwiGLU MLP (`has_mlp=True` for 'T', `has_mlp=False` for 't')
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int = 128,
        rotary_emb_dim: int = 64,
        window_size: int = -1,
        has_mlp: bool = True,
        mlp_expansion: float = 3.6,
    ):
        super().__init__()
        assert d_model % head_dim == 0, f"d_model ({d_model}) must be divisible by head_dim ({head_dim})"
        assert rotary_emb_dim <= head_dim, f"rotary_emb_dim ({rotary_emb_dim}) > head_dim ({head_dim})"
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.rotary_emb_dim = rotary_emb_dim
        self.window_size = window_size
        self.has_mlp = has_mlp

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(rotary_emb_dim) if rotary_emb_dim > 0 else None

        if has_mlp:
            self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
            self.mlp = SwiGLU(d_model, expansion=mlp_expansion)
        else:
            self.norm2 = None
            self.mlp = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCacheState:
        return KVCacheState(
            k=torch.empty(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            v=torch.empty(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(L, x.device, x.dtype)
            q, k = apply_partial_rope(q, k, cos, sin, self.rotary_emb_dim)

        if self.window_size > 0 and L > self.window_size + 1:
            # Sliding-window causal mask: queries at row i attend to keys j in [max(0, i - W), i]
            idx = torch.arange(L, device=x.device)
            diff = idx[:, None] - idx[None, :]
            attn_mask = (diff >= 0) & (diff <= self.window_size)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Fast FlashAttention-2 / cuDNN causal kernel
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn_out)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x: torch.Tensor, state: KVCacheState) -> torch.Tensor:
        """Single-step causal attention with KV cache for x of shape (B, 1, D)."""
        B, _, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = state.k.shape[2]
        if self.rope is not None:
            cos, sin = self.rope(1, x.device, x.dtype, offset=pos_offset)
            q, k = apply_partial_rope(q, k, cos, sin, self.rotary_emb_dim)

        state.k = torch.cat([state.k, k], dim=2)
        state.v = torch.cat([state.v, v], dim=2)

        if self.window_size > 0 and state.k.shape[2] > self.window_size + 1:
            k_win = state.k[:, :, -(self.window_size + 1) :, :]
            v_win = state.v[:, :, -(self.window_size + 1) :, :]
        else:
            k_win, v_win = state.k, state.v

        attn_out = F.scaled_dot_product_attention(q, k_win, v_win, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, D)

        x = x + self.out_proj(attn_out)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class IsotropicStack(nn.Module):
    """
    Parses a layout string like "m4", "T12", "T1m4", or "m4T1" (matching `goombalab/hnet/modules/isotropic.py`):
      - 'm': CausalSSMBlock(has_mlp=False)          -> adds 1 to residual height
      - 'M': CausalSSMBlock(has_mlp=True)           -> adds 2 to residual height
      - 't': CausalTransformerBlock(has_mlp=False)  -> adds 1 to residual height
      - 'T': CausalTransformerBlock(has_mlp=True)   -> adds 2 to residual height
    """

    def __init__(
        self,
        d_model: int,
        layout: str,
        head_dim: int = 128,
        rotary_emb_dim: int = 64,
        window_size: int = -1,
        conv_kernel_size: int = 4,
        mlp_expansion: float = 3.6,
    ):
        super().__init__()
        self.d_model = d_model
        self.layout = layout

        parsed = re.findall(r"([mMtT])(\d+)", layout)
        assert len(parsed) > 0, f"Invalid IsotropicStack layout string: {layout!r}"

        layers = []
        self.height = 0
        for arch_char, count_str in parsed:
            n_layers = int(count_str)
            has_mlp = arch_char.isupper()
            for _ in range(n_layers):
                if arch_char in ("m", "M"):
                    layers.append(
                        CausalSSMBlock(
                            d_model=d_model,
                            kernel_size=conv_kernel_size,
                            has_mlp=has_mlp,
                            mlp_expansion=mlp_expansion,
                        )
                    )
                else:
                    layers.append(
                        CausalTransformerBlock(
                            d_model=d_model,
                            head_dim=head_dim,
                            rotary_emb_dim=rotary_emb_dim,
                            window_size=window_size,
                            has_mlp=has_mlp,
                            mlp_expansion=mlp_expansion,
                        )
                    )
            self.height += n_layers * (2 if has_mlp else 1)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List:
        return [layer.init_state(batch_size, device, dtype) for layer in self.layers]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)

    def step(self, x: torch.Tensor, states: List) -> torch.Tensor:
        for layer, st in zip(self.layers, states):
            x = layer.step(x, st)
        return self.norm(x)
