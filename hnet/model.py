"""
Recursive Hierarchical Network (H-Net) and Byte-Level Language Model wrapper.
Matches the architecture of Cartesia / GoombaLab's official H-Net:
  - Pad/slice dimension expansion across nested stages
  - Zero-initialized fp32 residual projection around the chunk->main->dechunk bottleneck
  - Straight-Through Estimator (STE) residual combination
  - Per-stage learning rate multipliers
  - Byte-by-byte autoregressive generation (`generate`) with dynamic chunk skipping
  - Human-readable chunk visualization (`format_chunks`)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import IsotropicStack
from .config import HNetConfig
from .dc import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingOutput,
    RoutingState,
    ratio_loss,
    ste_func,
)


@dataclass
class HNetStageState:
    encoder_state: Optional[List] = None
    routing_state: Optional[RoutingState] = None
    main_state: Optional[Union["HNetStageState", List]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[List] = None


class HNetStage(nn.Module):
    """
    Recursive H-Net Stage.
    - If `stage_idx == config.num_stages` (innermost stage): wraps the core Causal Transformer.
    - Otherwise (outer stage): runs
        1. Encoder (SSM)
        2. RoutingModule + ChunkLayer
        3. Main Network (recursively another HNetStage)
        4. DeChunkLayer + STE Residual
        5. Decoder (SSM)
    """

    def __init__(self, config: HNetConfig, stage_idx: int = 0):
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        self.is_innermost = stage_idx == config.num_stages

        # Learned zero-initialized padding if this stage is wider than its parent stage
        if stage_idx > 0 and self.d_model > config.d_model[stage_idx - 1]:
            diff = self.d_model - config.d_model[stage_idx - 1]
            self.pad_dimension = nn.Parameter(torch.zeros(diff))
        else:
            self.pad_dimension = None

        if self.is_innermost:
            self.main_network = IsotropicStack(
                d_model=self.d_model,
                n_layers=config.n_main_layers,
                block_type="transformer",
                head_dim=config.head_dim,
            )
        else:
            self.encoder = IsotropicStack(
                d_model=self.d_model,
                n_layers=config.n_encoder_layers[stage_idx],
                block_type="ssm",
                conv_kernel_size=config.conv_kernel_size,
            )
            self.routing_module = RoutingModule(self.d_model)
            self.chunk_layer = ChunkLayer()
            self.main_network = HNetStage(config, stage_idx=stage_idx + 1)
            self.dechunk_layer = DeChunkLayer(self.d_model)
            self.decoder = IsotropicStack(
                d_model=self.d_model,
                n_layers=config.n_decoder_layers[stage_idx],
                block_type="ssm",
                conv_kernel_size=config.conv_kernel_size,
            )

            # Zero-initialized fp32 residual projection (crucial for initial gradient flow)
            self.residual_proj = nn.Linear(self.d_model, self.d_model, dtype=torch.float32)
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True

        # Tag parameters with stage_idx for per-stage learning rate multipliers
        for p in self.parameters(recurse=False):
            p._stage_idx = stage_idx
        if self.is_innermost:
            for p in self.main_network.parameters():
                p._stage_idx = stage_idx
        else:
            for mod in (self.encoder, self.routing_module, self.dechunk_layer, self.decoder, self.residual_proj):
                for p in mod.parameters():
                    p._stage_idx = stage_idx

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> HNetStageState:
        if self.is_innermost:
            return HNetStageState(
                main_state=self.main_network.init_state(batch_size, device, dtype)
            )
        return HNetStageState(
            encoder_state=self.encoder.init_state(batch_size, device, dtype),
            routing_state=self.routing_module.init_state(batch_size, device, dtype),
            main_state=self.main_network.init_state(batch_size, device, dtype),
            dechunk_state=self.dechunk_layer.init_state(batch_size, device, dtype),
            decoder_state=self.decoder.init_state(batch_size, device, dtype),
        )

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[RoutingOutput, torch.Tensor]]]:
        in_dim = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            pad = self.pad_dimension.to(hidden_states.dtype).expand(*hidden_states.shape[:-1], -1)
            hidden_states = torch.cat([hidden_states, pad], dim=-1)

        if self.is_innermost:
            out = self.main_network(hidden_states, mask=mask)
            return out[..., :in_dim], []

        # 1. Encoder (fine resolution L)
        h_enc = self.encoder(hidden_states, mask=mask)
        residual = self.residual_proj(h_enc.float())

        # 2. Dynamic Chunker: route & downsample (B, L, D) -> (B, M, D)
        bpred = self.routing_module(h_enc, mask=mask)
        h_chunked, next_mask, sorted_indices = self.chunk_layer(h_enc, bpred.boundary_mask)

        # 3. Inner Main Network (runs on M compressed chunks)
        h_inner, inner_routes = self.main_network(h_chunked, mask=next_mask)

        # 4. Dechunker: EMA smoothing + broadcast back to L + STE residual
        h_dechunk = self.dechunk_layer(
            h_inner, bpred.boundary_mask, bpred.boundary_prob, sorted_indices
        )
        h_combined = (
            h_dechunk.float() * ste_func(bpred.selected_probs.float()) + residual
        ).to(h_enc.dtype)

        # 5. Decoder (fine resolution L)
        out = self.decoder(h_combined, mask=mask)
        return out[..., :in_dim], [(bpred, mask), *inner_routes]

    def step(
        self, hidden_state: torch.Tensor, state: HNetStageState
    ) -> Tuple[torch.Tensor, List[RoutingOutput]]:
        """
        Single-byte autoregressive step for batch_size == 1 (or B where all items share boundary).
        If boundary_mask is False, the inner main_network is completely skipped (0 FLOPs)!
        """
        in_dim = hidden_state.shape[-1]
        if self.pad_dimension is not None:
            pad = self.pad_dimension.to(hidden_state.dtype).expand(*hidden_state.shape[:-1], -1)
            hidden_state = torch.cat([hidden_state, pad], dim=-1)

        if self.is_innermost:
            out = self.main_network.step(hidden_state, state.main_state)
            return out[..., :in_dim], []

        # 1. Encoder always steps on every byte
        h_enc = self.encoder.step(hidden_state, state.encoder_state)
        residual = self.residual_proj(h_enc.float())

        # 2. Router predicts whether this byte starts a new chunk
        bpred = self.routing_module.step(h_enc, state.routing_state)

        # 3. Main network ONLY runs if boundary_mask is True
        if bpred.boundary_mask.any():
            h_inner_in = h_enc[bpred.boundary_mask]  # (B_sel, 1, D)
            h_inner_out, inner_routes = self.main_network.step(h_inner_in, state.main_state)
        else:
            h_inner_out = torch.empty(0, device=h_enc.device, dtype=h_enc.dtype)
            inner_routes = []

        # 4. Dechunker updates EMA cache if boundary_mask is True, else reuses cached chunk vector
        h_dechunk = self.dechunk_layer.step(
            h_inner_out, bpred.boundary_mask, bpred.boundary_prob, state.dechunk_state
        )
        h_combined = (
            h_dechunk.float() * ste_func(bpred.selected_probs.unsqueeze(1).float()) + residual
        ).to(h_enc.dtype)

        # 5. Decoder steps on every byte to predict the next byte
        out = self.decoder.step(h_combined, state.decoder_state)
        return out[..., :in_dim], [bpred, *inner_routes]


class ByteHNetLM(nn.Module):
    """
    End-to-End Byte-Level H-Net Language Model.
    Takes raw UTF-8 byte IDs in [0..255], learns dynamic hierarchical chunking end-to-end,
    and outputs next-byte logits along with auxiliary compression ratio loss.
    """

    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        d0 = config.d_model[0]

        self.byte_embedding = nn.Embedding(config.vocab_size, d0)
        self.backbone = HNetStage(config, stage_idx=0)
        self.lm_head = nn.Linear(d0, config.vocab_size, bias=False)

        self._init_weights()
        for p in self.byte_embedding.parameters():
            p._stage_idx = 0
        for p in self.lm_head.parameters():
            p._stage_idx = 0

    def _init_weights(self, init_std: float = 0.02):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)

    def get_param_groups(self, base_lr: float, weight_decay: float = 0.1) -> List[Dict]:
        """
        Creates optimizer parameter groups with per-stage learning rate multipliers
        (outer stages receive a higher LR multiplier since they process more tokens per step).
        """
        groups: Dict[Tuple[int, float], List[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            stage_idx = getattr(param, "_stage_idx", 0)
            no_decay = param.ndim < 2 or "norm" in name or "bias" in name or "pad_dimension" in name
            wd = 0.0 if no_decay else weight_decay
            key = (stage_idx, wd)
            groups.setdefault(key, []).append(param)

        param_groups = []
        for (stage_idx, wd), params in groups.items():
            mult = self.config.lr_multipliers[min(stage_idx, len(self.config.lr_multipliers) - 1)]
            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr * mult,
                    "lr_mult": mult,
                    "weight_decay": wd,
                    "stage_idx": stage_idx,
                }
            )
        return param_groups

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        if mask is None:
            mask = torch.ones(B, L, device=input_ids.device, dtype=torch.bool)

        x = self.byte_embedding(input_ids)
        hidden, route_list = self.backbone(x, mask=mask)
        logits = self.lm_head(hidden)

        result: Dict[str, Union[torch.Tensor, List]] = {
            "logits": logits,
            "routes": route_list,
        }

        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
            bpb = ce_loss / 0.6931471805599453  # nats -> bits per byte (ln 2)

            aux_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            compression_stats = []
            for stage_i, (r_out, r_mask) in enumerate(route_list):
                target_N = self.config.target_compression[stage_i]
                aux_loss = aux_loss + ratio_loss(r_out, r_mask, target_compression_N=target_N)
                valid_tokens = r_mask.float().sum().clamp(min=1.0)
                selected_chunks = (r_out.boundary_mask & r_mask).float().sum().clamp(min=1.0)
                compression_stats.append((valid_tokens / selected_chunks).item())

            total_loss = ce_loss + self.config.aux_loss_weight * aux_loss
            result.update(
                {
                    "loss": total_loss,
                    "ce_loss": ce_loss,
                    "bpb": bpb,
                    "aux_loss": aux_loss,
                    "compression_ratios": compression_stats,
                }
            )

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_bytes: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Autoregressive byte-by-byte generation using the recurrent `step()` cache.
        Returns:
            generated_text: decoded UTF-8 string
            chunked_view: string with '|' showing where Stage 0 dynamically placed chunk boundaries
            stats: dict with empirical compression ratio and inner-network skip percentage
        """
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        prompt_bytes = list(prompt.encode("utf-8"))
        if not prompt_bytes:
            prompt_bytes = [ord(" ")]

        state = self.backbone.init_state(batch_size=1, device=device, dtype=dtype)
        all_bytes: List[int] = []
        stage0_boundaries: List[bool] = []
        inner_steps_taken = 0

        # 1. Prefill prompt byte-by-byte through `step` so recurrent + KV states are exact
        last_logits = None
        for b_val in prompt_bytes:
            inp = torch.tensor([[b_val]], device=device, dtype=torch.long)
            emb = self.byte_embedding(inp)
            out_h, routes = self.backbone.step(emb, state)
            last_logits = self.lm_head(out_h[:, -1, :])

            is_boundary = bool(routes[0].boundary_mask.item())
            all_bytes.append(b_val)
            stage0_boundaries.append(is_boundary)
            if is_boundary:
                inner_steps_taken += 1

        # 2. Autoregressively sample new bytes
        for _ in range(max_new_bytes):
            logits = last_logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_byte = int(torch.multinomial(probs, num_samples=1).item())

            inp = torch.tensor([[next_byte]], device=device, dtype=torch.long)
            emb = self.byte_embedding(inp)
            out_h, routes = self.backbone.step(emb, state)
            last_logits = self.lm_head(out_h[:, -1, :])

            is_boundary = bool(routes[0].boundary_mask.item())
            all_bytes.append(next_byte)
            stage0_boundaries.append(is_boundary)
            if is_boundary:
                inner_steps_taken += 1

        full_text = bytes(all_bytes).decode("utf-8", errors="replace")

        # Format visual chunk boundaries: insert '|' wherever Stage 0 triggered a new chunk
        pieces: List[str] = []
        current_chunk: List[int] = []
        for idx, (b_val, is_b) in enumerate(zip(all_bytes, stage0_boundaries)):
            if is_b and idx > 0:
                pieces.append(bytes(current_chunk).decode("utf-8", errors="replace"))
                current_chunk = [b_val]
            else:
                current_chunk.append(b_val)
        if current_chunk:
            pieces.append(bytes(current_chunk).decode("utf-8", errors="replace"))
        chunked_view = "|".join(pieces)

        total_steps = len(all_bytes)
        stats = {
            "total_bytes": total_steps,
            "stage0_chunks": inner_steps_taken,
            "bytes_per_chunk": total_steps / max(inner_steps_taken, 1),
            "main_net_skip_pct": 100.0 * (1.0 - inner_steps_taken / max(total_steps, 1)),
        }
        return full_text, chunked_view, stats
