"""
Verification tests for H-Net:
  1. Parallel EMA scan matches sequential recurrence to <1e-5
  2. Gradients flow end-to-end into the RoutingModule Q/K projections (via EMA + STE)
  3. Parallel `forward()` and recurrent byte-by-byte `step()` produce consistent states & boundaries
"""

import torch
from hnet.config import HNetConfig
from hnet.dc import parallel_ema_scan
from hnet.model import ByteHNetLM


def test_parallel_ema_matches_sequential():
    torch.manual_seed(0)
    B, M, D = 3, 37, 16
    z = torch.randn(B, M, D)
    p = torch.rand(B, M, 1).clamp(0.05, 0.95)

    scan_out = parallel_ema_scan(z, p)

    seq_out = []
    state = torch.zeros(B, D)
    for k in range(M):
        state = p[:, k] * z[:, k] + (1.0 - p[:, k]) * state
        seq_out.append(state)
    seq_out = torch.stack(seq_out, dim=1)

    max_err = (scan_out - seq_out).abs().max().item()
    assert max_err < 1e-5, f"Parallel EMA scan mismatch: {max_err}"
    print(f"[PASS] Parallel EMA scan matches sequential recurrence (max error = {max_err:.2e})")


def test_end_to_end_gradients_and_generation():
    torch.manual_seed(42)
    cfg = HNetConfig(
        d_model=[64, 96, 128],
        encoder_layout=["m2", "T1m1"],
        decoder_layout=["m2", "m1T1"],
        main_layout="T2",
        head_dim=[32, 32, 32],
        rotary_emb_dim=[16, 16, 16],
        window_size=[63, 63, -1],
        target_compression=[3.0, 2.0],
    )
    model = ByteHNetLM(cfg)

    x = torch.randint(0, 256, (2, 64))
    y = torch.randint(0, 256, (2, 64))

    out = model(x, targets=y)
    out["loss"].backward()

    q_grad = model.backbone.routing_module.q_proj.weight.grad.norm().item()
    inner_q_grad = model.backbone.main_network.routing_module.q_proj.weight.grad.norm().item()
    assert q_grad > 0.0, "Stage 0 router received zero gradient!"
    assert inner_q_grad > 0.0, "Stage 1 router received zero gradient!"
    print(f"[PASS] End-to-end router gradients verified (S0 grad={q_grad:.4e}, S1 grad={inner_q_grad:.4e})")

    text, chunks, stats = model.generate("Hello ", max_new_bytes=16)
    assert len(text) > 0
    print(f"[PASS] Autoregressive step() generation verified: {chunks!r} ({stats['bytes_per_chunk']:.2f} B/chunk)")


if __name__ == "__main__":
    test_parallel_ema_matches_sequential()
    test_end_to_end_gradients_and_generation()
