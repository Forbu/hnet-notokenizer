import torch
from hnet.dit2d import HNetDiT2D, HNetDiT2DConfig
from hnet.hilbert import build_hilbert_lut


def test_hilbert_locality_and_bijection():
    for grid_size in (8, 16, 32, 64):
        h2r, r2h, pos_norm, pos_grid = build_hilbert_lut(grid_size)
        L = grid_size * grid_size
        assert h2r.shape == (L,)
        assert r2h.shape == (L,)

        # Bijection check
        arange = torch.arange(L)
        assert torch.equal(h2r[r2h], arange)
        assert torch.equal(r2h[h2r], arange)

        # 4-connected 2D spatial locality check: every step t -> t+1 has Manhattan distance == 1
        diffs = (pos_grid[1:] - pos_grid[:-1]).abs().sum(dim=-1)
        assert torch.all(diffs == 1.0), f"Hilbert locality violated for grid_size={grid_size}"


def test_patchify_unpatchify_roundtrip():
    cfg = HNetDiT2DConfig.dit2d_small(patch_size=2)
    model = HNetDiT2D(cfg)
    img = torch.randn(4, 3, 64, 64)
    patches = model.patchify_hilbert(img)
    assert patches.shape == (4, 1024, 12)
    recon = model.unpatchify_hilbert(patches)
    assert torch.allclose(img, recon, atol=1e-6)


def test_dit2d_forward_backward_and_routing_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HNetDiT2DConfig(
        img_size=64,
        patch_size=2,
        d_model=[128, 256],
        d_cond=128,
        num_encoder_blocks=2,
        num_decoder_blocks=2,
        num_main_blocks=2,
        head_dim=64,
        rotary_emb_dim=32,
        target_compression=4.0,
        routing_mode="dynamic",
    )
    model = HNetDiT2D(cfg).to(device)
    x_1 = torch.randn(2, 3, 64, 64, device=device)
    y = torch.tensor([10, 42], device=device)

    loss, metrics = model.forward_loss(x_1, y)
    assert torch.isfinite(loss)
    loss.backward()

    # Verify router q_proj and k_proj receive non-zero gradients from both CE/flow STE and ratio_loss
    assert model.routing_module.q_proj.weight.grad is not None
    assert model.routing_module.k_proj.weight.grad is not None
    assert model.routing_module.q_proj.weight.grad.abs().sum().item() > 0.0
    assert model.routing_module.k_proj.weight.grad.abs().sum().item() > 0.0
    print("Dynamic 2D H-Net-DiT test metrics:", metrics)


def test_dit2d_uniform_mode_and_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HNetDiT2DConfig(
        img_size=64,
        patch_size=2,
        d_model=[128, 256],
        d_cond=128,
        num_encoder_blocks=2,
        num_decoder_blocks=2,
        num_main_blocks=2,
        head_dim=64,
        rotary_emb_dim=32,
        target_compression=4.0,
        routing_mode="uniform",
    )
    model = HNetDiT2D(cfg).to(device)
    x_1 = torch.randn(2, 3, 64, 64, device=device)
    y = torch.tensor([1, 2], device=device)
    loss, metrics = model.forward_loss(x_1, y)
    assert abs(metrics["compression_ratio"] - 4.0) < 1e-4

    samples, traj = model.sample(
        batch_size=2, y=y, steps=4, cfg_scale=1.5, device=device, return_trajectories=True
    )
    assert samples.shape == (2, 3, 64, 64)
    assert len(traj) > 0
    assert traj[-1]["boundary_2d"].shape == (2, 32, 32)
    assert traj[-1]["chunk_id_2d"].shape == (2, 32, 32)


if __name__ == "__main__":
    test_hilbert_locality_and_bijection()
    test_patchify_unpatchify_roundtrip()
    test_dit2d_forward_backward_and_routing_gradients()
    test_dit2d_uniform_mode_and_sampling()
    print("All 2D H-Net-DiT tests passed!")
