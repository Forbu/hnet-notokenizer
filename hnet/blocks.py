"""
Backbone sequence blocks for H-Net:
  1. CausalSSMBlock: Mamba-inspired Gated Causal Conv1d + Parallel Linear Recurrence + SwiGLU
     for outer Encoder/Decoder stages (O(L) complexity, strong byte-compression inductive bias).
  2. CausalTransformerBlock: RoPE + FlashAttention (SDPA) + SwiGLU for the innermost Main Network.
"""

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
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0):
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        hidden = int(8 * d_model / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@dataclass
class SSMLayerState:
    conv_buf: torch.Tensor  # (B, D_inner, K-1)
    ssm_state: torch.Tensor # (B, D_inner)


@dataclass
class KVCacheState:
    k: torch.Tensor         # (B, H, L_cur, Dh)
    v: torch.Tensor         # (B, H, L_cur, Dh)


class CausalSSMBlock(nn.Module):
    """
    Lightweight Mamba-style State Space / Gated Recurrence Block for outer Encoder & Decoder stages.
    Combines:
      1. Causal depthwise 1D convolution (captures local byte/character n-grams)
      2. Data-dependent forget-gate linear recurrence (parallel O(log L) scan in training, O(1) in step)
      3. SwiGLU channel mixer
    """

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.norm1 = nn.RMSNorm(d_model)
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

        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

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
        p_forget = torch.sigmoid(forget_logits.float()).mean(dim=-1, keepdim=True)  # (B, L, 1)
        if mask is not None:
            p_forget = p_forget * mask.unsqueeze(-1).float()
        rec_out = parallel_ema_scan(u_conv.float(), p_forget).to(x.dtype)

        # 3. Output gating & projection
        x = x + self.out_proj(rec_out * F.silu(gate))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x: torch.Tensor, state: SSMLayerState) -> torch.Tensor:
        """O(1) single-step update for x of shape (B, 1, D)."""
        h = self.norm1(x)
        u, gate, forget_logits = self.in_proj(h).chunk(3, dim=-1)  # each (B, 1, D)

        # Update causal conv buffer of length K
        u_col = u.transpose(1, 2)  # (B, D, 1)
        conv_in = torch.cat([state.conv_buf, u_col], dim=-1)  # (B, D, K)
        state.conv_buf.copy_(conv_in[:, :, 1:])
        weight = self.conv1d.weight.squeeze(1)  # (D, K)
        u_conv = (conv_in * weight.unsqueeze(0)).sum(dim=-1) + self.conv1d.bias  # (B, D)
        u_conv = F.silu(u_conv)

        # Update recurrent state
        p_forget = torch.sigmoid(forget_logits.squeeze(1).float()).mean(dim=-1, keepdim=True).to(x.dtype)
        new_ssm = p_forget * u_conv + (1.0 - p_forget) * state.ssm_state
        state.ssm_state.copy_(new_ssm)

        out = self.out_proj(new_ssm.unsqueeze(1) * F.silu(gate))
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class CausalTransformerBlock(nn.Module):
    """
    Causal Transformer block (RoPE + FlashAttention SDPA + SwiGLU) for the innermost Main Network.
    """

    def __init__(self, d_model: int, head_dim: int = 64):
        super().__init__()
        assert d_model % head_dim == 0, f"d_model ({d_model}) must be divisible by head_dim ({head_dim})"
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim

        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim)

        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

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

        cos, sin = self.rope(L, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Uses FlashAttention-2 / Memory-Efficient kernel automatically on H100 when is_causal=True
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

    def step(
        self, x: torch.Tensor, state: KVCacheState, active_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single-step causal attention with KV cache for x of shape (B, 1, D)."""
        B, _, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = state.k.shape[2]
        cos, sin = self.rope(1, x.device, x.dtype, offset=pos_offset)
        q, k = apply_rope(q, k, cos, sin)

        state.k = torch.cat([state.k, k], dim=2)
        state.v = torch.cat([state.v, v], dim=2)

        attn_out = F.scaled_dot_product_attention(q, state.k, state.v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, D)

        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class IsotropicStack(nn.Module):
    """
    A sequence of either CausalSSMBlocks (for outer Encoder/Decoder)
    or CausalTransformerBlocks (for the innermost Main Network), ending with RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        block_type: str = "ssm",
        head_dim: int = 64,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_type = block_type
        if block_type == "ssm":
            self.layers = nn.ModuleList(
                [CausalSSMBlock(d_model, kernel_size=conv_kernel_size) for _ in range(n_layers)]
            )
        elif block_type == "transformer":
            self.layers = nn.ModuleList(
                [CausalTransformerBlock(d_model, head_dim=head_dim) for _ in range(n_layers)]
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        self.norm = nn.RMSNorm(d_model)

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
