"""
Dynamic Chunking (DC) primitives for H-Net:
  1. STE (Straight-Through Estimator)
  2. RoutingModule: predicts chunk boundaries via cosine dissimilarity of adjacent states
  3. ChunkLayer: packs boundary states from (B, L, D) -> (B, M, D)
  4. DeChunkLayer: confidence-weighted EMA smoothing + causal broadcast back to (B, L, D)
  5. ratio_loss: Switch-Transformer-style auxiliary load-balancing loss targeting compression N
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _ema_fwd_kernel(
        Z_ptr,
        P_ptr,
        Out_ptr,
        L,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        z_base = Z_ptr + b * L * D + offs_d
        o_base = Out_ptr + b * L * D + offs_d
        p_base = P_ptr + b * L

        state = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t in range(L):
            p_t = tl.load(p_base + t).to(tl.float32)
            z_t = tl.load(z_base + t * D, mask=mask_d, other=0.0).to(tl.float32)
            state = p_t * z_t + (1.0 - p_t) * state
            tl.store(o_base + t * D, state.to(Out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def _ema_bwd_kernel(
        Z_ptr,
        P_ptr,
        Out_ptr,
        GradOut_ptr,
        GradZ_ptr,
        GradPPart_ptr,
        n_tiles_d,
        L,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        z_base = Z_ptr + b * L * D + offs_d
        o_base = Out_ptr + b * L * D + offs_d
        go_base = GradOut_ptr + b * L * D + offs_d
        gz_base = GradZ_ptr + b * L * D + offs_d
        p_base = P_ptr + b * L
        gp_base = GradPPart_ptr + (b * n_tiles_d + pid_d) * L

        grad_state = tl.zeros([BLOCK_D], dtype=tl.float32)
        p_next = 0.0
        for step in range(L):
            t = L - 1 - step
            go_t = tl.load(go_base + t * D, mask=mask_d, other=0.0).to(tl.float32)
            p_t = tl.load(p_base + t).to(tl.float32)
            z_t = tl.load(z_base + t * D, mask=mask_d, other=0.0).to(tl.float32)
            out_prev = tl.load(
                o_base + (t - 1) * D, mask=mask_d & (t > 0), other=0.0
            ).to(tl.float32)

            grad_state = go_t + (1.0 - p_next) * grad_state
            gz_t = p_t * grad_state
            gp_t = tl.sum(grad_state * (z_t - out_prev), axis=0)

            tl.store(gz_base + t * D, gz_t.to(GradZ_ptr.dtype.element_ty), mask=mask_d)
            tl.store(gp_base + t, gp_t)
            p_next = p_t


class _TritonEMAScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        z_c = z.contiguous()
        p_c = p.squeeze(-1).contiguous()
        B, L, D = z_c.shape
        out = torch.empty_like(z_c)
        if L == 0:
            ctx.save_for_backward(z_c, p_c, out)
            return out
        BLOCK_D = min(128, triton.next_power_of_2(D))
        grid = (B, triton.cdiv(D, BLOCK_D))
        _ema_fwd_kernel[grid](z_c, p_c, out, L, D=D, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(z_c, p_c, out)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z_c, p_c, out = ctx.saved_tensors
        B, L, D = z_c.shape
        if L == 0:
            return torch.zeros_like(z_c), torch.zeros_like(p_c).unsqueeze(-1)
        grad_out_c = grad_out.contiguous()
        grad_z = torch.empty_like(z_c)
        BLOCK_D = ctx.BLOCK_D
        n_tiles_d = triton.cdiv(D, BLOCK_D)
        grad_p_part = torch.empty((B, n_tiles_d, L), device=z_c.device, dtype=torch.float32)
        grid = (B, n_tiles_d)
        _ema_bwd_kernel[grid](
            z_c, p_c, out, grad_out_c, grad_z, grad_p_part, n_tiles_d, L, D=D, BLOCK_D=BLOCK_D
        )
        grad_p = grad_p_part.sum(dim=1).unsqueeze(-1).to(p_c.dtype)
        return grad_z, grad_p


class STE(torch.autograd.Function):
    """Forward: returns 1.0. Backward: passes gradient straight through to input probability."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def ste_func(x: torch.Tensor) -> torch.Tensor:
    return STE.apply(x)


def parallel_ema_scan(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Computes the first-order recurrence across chunks k = 0 .. M-1 in O(log M) parallel GPU steps:
        ema_k = p_k * z_k + (1 - p_k) * ema_{k-1},   with ema_{-1} = 0

    Args:
        z: (B, M, D) float32 chunk representations
        p: (B, M, 1) float32 boundary probabilities in (0, 1)
    Returns:
        ema: (B, M, D) float32 smoothed chunk representations
    """
    B, M, D = z.shape
    if M == 0:
        return z

    if HAS_TRITON and z.is_cuda:
        return _TritonEMAScan.apply(z, p)

    a = 1.0 - p          # (B, M, 1) decay factor
    b = p * z            # (B, M, D) weighted input

    # Hillis-Steele associative scan in O(log2(M)) out-of-place steps (autograd-safe)
    offset = 1
    while offset < M:
        b_new = a[:, offset:] * b[:, :-offset] + b[:, offset:]
        a_new = a[:, offset:] * a[:, :-offset]
        b = torch.cat([b[:, :offset], b_new], dim=1)
        a = torch.cat([a[:, :offset], a_new], dim=1)
        offset <<= 1

    return b


@dataclass
class RoutingOutput:
    boundary_prob: torch.Tensor   # (B, L, 2) -> [..., 0]=P(merge), [..., 1]=P(boundary)
    boundary_mask: torch.Tensor   # (B, L) bool -> True where a new chunk begins
    selected_probs: torch.Tensor  # (B, L, 1) -> router confidence of the selected action


@dataclass
class RoutingState:
    has_seen_tokens: torch.Tensor    # (B,) bool
    last_hidden_state: torch.Tensor  # (B, D)


@dataclass
class DeChunkState:
    last_value: torch.Tensor         # (B, D)


class RoutingModule(nn.Module):
    """
    Predicts dynamic chunk boundaries by comparing adjacent hidden states (t-1, t)
    via cosine similarity after learned Q and K projections.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RoutingState:
        return RoutingState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> RoutingOutput:
        """
        Args:
            hidden_states: (B, L, D)
            mask: (B, L) bool mask of valid positions
        """
        q = F.normalize(self.q_proj(hidden_states[:, :-1]), dim=-1)
        k = F.normalize(self.k_proj(hidden_states[:, 1:]), dim=-1)

        cos_sim = torch.einsum("bld,bld->bl", q, k)
        p_boundary = torch.clamp((1.0 - cos_sim) / 2.0, min=0.0, max=1.0)

        # First token in every sequence is forced to be a boundary (p_0 = 1.0)
        p_boundary = F.pad(p_boundary, (1, 0), value=1.0)

        boundary_prob = torch.stack([1.0 - p_boundary, p_boundary], dim=-1)  # (B, L, 2)
        selected_idx = torch.argmax(boundary_prob, dim=-1)                   # (B, L)
        boundary_mask = (selected_idx == 1) & mask
        # Guarantee at least position 0 is a boundary for every batch row
        boundary_mask[:, 0] = True

        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))

        return RoutingOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(self, hidden_state: torch.Tensor, state: RoutingState) -> RoutingOutput:
        """
        Single-step autoregressive routing for hidden_state of shape (B, 1, D).
        """
        h = hidden_state.squeeze(1)  # (B, D)
        q = F.normalize(self.q_proj(state.last_hidden_state), dim=-1)
        k = F.normalize(self.k_proj(h), dim=-1)

        cos_sim = torch.einsum("bd,bd->b", q, k)
        p_boundary = torch.clamp((1.0 - cos_sim) / 2.0, min=0.0, max=1.0)

        # First token is always a boundary
        p_boundary = torch.where(
            state.has_seen_tokens, p_boundary, torch.ones_like(p_boundary)
        )
        state.last_hidden_state.copy_(h)
        state.has_seen_tokens.fill_(True)

        boundary_prob = torch.stack([1.0 - p_boundary, p_boundary], dim=-1)  # (B, 2)
        boundary_mask = boundary_prob[..., 1] > 0.5                          # (B,)
        selected_probs = boundary_prob.max(dim=-1).values.unsqueeze(-1)      # (B, 1)

        return RoutingOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )


class ChunkLayer(nn.Module):
    """
    Downsamples (B, L, D) -> (B, M, D) by gathering boundary positions per batch item.
    """

    def forward(
        self, hidden_states: torch.Tensor, boundary_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = hidden_states.shape
        device = hidden_states.device

        num_chunks = boundary_mask.sum(dim=-1)          # (B,)
        max_chunks = max(int(num_chunks.max().item()), 1)

        # Stable sort: valid boundary positions get indices [0..L-1], non-boundaries get [L..2L-1]
        token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        sorted_indices = torch.argsort(token_idx, dim=1)[:, :max_chunks]  # (B, M)

        chunked_states = torch.gather(
            hidden_states,
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(B, max_chunks, D),
        )
        next_mask = torch.arange(max_chunks, device=device)[None, :] < num_chunks[:, None]

        return chunked_states, next_mask, sorted_indices


class DeChunkLayer(nn.Module):
    """
    1. Smooths compressed chunk vectors via confidence-weighted EMA:
         z_tilde_k = p_k * z_k + (1 - p_k) * z_tilde_{k-1}
    2. Causally broadcasts each chunk vector across all positions up to the next boundary.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DeChunkState:
        return DeChunkState(
            last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        )

    def forward(
        self,
        chunk_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, M, D = chunk_states.shape
        L = boundary_mask.shape[1]
        orig_dtype = chunk_states.dtype

        # Gather router boundary probabilities p_k for the M selected chunks
        p_all = torch.clamp(boundary_prob[..., 1].float(), min=1e-4, max=1.0 - 1e-4)
        p_chunk = torch.gather(p_all, dim=1, index=sorted_indices).unsqueeze(-1)  # (B, M, 1)

        # Parallel associative EMA scan over the M chunks
        smoothed = parallel_ema_scan(chunk_states.float(), p_chunk).to(orig_dtype)  # (B, M, D)

        # Plug back into original length L via cumulative sum of boundary_mask
        plug_back_idx = torch.clamp(torch.cumsum(boundary_mask.long(), dim=1) - 1, min=0, max=M - 1)
        dechunked = torch.gather(
            smoothed,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(B, L, D),
        )
        return dechunked

    def step(
        self,
        chunk_state: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        state: DeChunkState,
    ) -> torch.Tensor:
        """
        Single-step dechunking during autoregressive generation.
        - If boundary_mask is True: updates state.last_value via EMA with new chunk_state.
        - If boundary_mask is False: reuses state.last_value without running main_network!
        """
        B = boundary_mask.shape[0]
        D = self.d_model
        dtype = state.last_value.dtype
        device = state.last_value.device

        p = torch.zeros(B, 1, device=device, dtype=dtype)
        p[boundary_mask, 0] = boundary_prob[boundary_mask, 1].clamp(min=1e-4, max=1.0 - 1e-4).to(dtype)

        current = torch.zeros(B, D, device=device, dtype=dtype)
        if chunk_state.numel() > 0:
            current[boundary_mask] = chunk_state.squeeze(1).to(dtype)

        # When boundary_mask is False, p == 0 -> result == state.last_value
        result = p * current + (1.0 - p) * state.last_value
        state.last_value.copy_(result)
        return result.unsqueeze(1)  # (B, 1, D)


def ratio_loss(
    router_output: RoutingOutput,
    mask: torch.Tensor,
    target_compression_N: float,
) -> torch.Tensor:
    """
    Switch-Transformer-style auxiliary loss driving the boundary rate toward 1 / N.
    Achieves its global minimum of 1.0 when true_ratio == avg_prob == 1 / N.
    """
    valid = mask.float()
    denom = valid.sum().clamp(min=1.0)

    true_ratio = (router_output.boundary_mask.float() * valid).sum() / denom
    avg_prob = (router_output.boundary_prob[..., 1].float() * valid).sum() / denom

    N = target_compression_N
    return ((1.0 - true_ratio) * (1.0 - avg_prob) + true_ratio * avg_prob * (N - 1.0)) * (
        N / (N - 1.0)
    )
