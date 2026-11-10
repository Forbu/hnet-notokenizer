"""
Quantitative & Visual Analysis of Emergent 2D Dynamic Chunking in Bidirectional 2D H-Net-DiT
on ImageNet-64 (`Forbu14/imagenet-64`).

Produces:
  1. Quantitative Adaptive Compute Table: measures router boundary probability p_t, boundary rate b_t,
     and effective chunk size s_k across quartiles of local 2D image detail (local patch variance / gradient).
  2. Multi-panel figure (`figures/dit2d_val_chunk_analysis.png`):
     GT Image | Predicted x_1 | Router Boundary Prob p_t (2D) | Dynamic 2D Chunk Borders | Chunk Size Map s_k
  3. ODE Trajectory figure (`figures/dit2d_ode_trajectory_chunks.png`):
     Evolution of x_t, x1_pred, and 2D dynamic chunk boundaries from t=0.0 -> t=1.0 during Rectified Flow sampling.
"""

import argparse
import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from hnet.dit2d import HNetDiT2D, HNetDiT2DConfig
from train_dit2d import render_chunk_contours_on_image


def load_checkpoint(ckpt_path: str, device: torch.device) -> HNetDiT2D:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = HNetDiT2DConfig(**ckpt["config"])
    model = HNetDiT2D(cfg).to(device)
    state = {k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def analyze_entropy_vs_chunking(
    model: HNetDiT2D,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    t_val: float = 0.75,
    num_samples: int = 512,
) -> Dict[str, float]:
    """
    Correlate local 2D image complexity (Sobel edge energy + patch standard deviation per 2x2 micro-patch)
    with router boundary probability p_t, hard boundary rate b_t, and chunk size s_k.
    """
    p = model.config.patch_size
    G = model.config.grid_size

    all_detail = []
    all_prob = []
    all_bmask = []
    all_csize = []

    gen = torch.Generator(device=device).manual_seed(777)
    batch_size = 64

    for start in range(0, min(num_samples, val_images.shape[0]), batch_size):
        end = min(start + batch_size, num_samples)
        B = end - start
        x_1 = val_images[start:end].to(device).float() / 127.5 - 1.0
        y = val_labels[start:end].to(device)
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        noise = torch.randn(x_1.shape, device=device, generator=gen)
        x_t = t_val * x_1 + (1.0 - t_val) * noise

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, router_out, chunk_out = model(x_t, t, y)

        # Compute local 2D spatial gradient energy at each (G, G) micro-patch on clean x_1
        # 1. Within-patch standard deviation + 2. Spatial finite difference across adjacent micro-patches
        patches_2d = (
            x_1.view(B, 3, G, p, G, p).permute(0, 2, 4, 1, 3, 5).reshape(B, G, G, 3 * p * p)
        )
        patch_mean_rgb = patches_2d.view(B, G, G, 3, p * p).mean(dim=-1)  # (B, G, G, 3)
        patch_std = patches_2d.std(dim=-1)                                # (B, G, G)

        dx = F.pad((patch_mean_rgb[:, :, 1:, :] - patch_mean_rgb[:, :, :-1, :]).norm(dim=-1), (1, 0))
        dy = F.pad((patch_mean_rgb[:, 1:, :, :] - patch_mean_rgb[:, :-1, :, :]).norm(dim=-1), (0, 0, 1, 0))
        local_detail_2d = patch_std + 0.5 * (dx + dy)                     # (B, G, G)

        # Map router outputs from Hilbert (B, L) -> 2D raster (B, G, G)
        p_2d = router_out.boundary_prob[..., 1].float().index_select(1, model.raster_to_hilbert).view(B, G, G)
        b_2d = router_out.boundary_mask.float().index_select(1, model.raster_to_hilbert).view(B, G, G)

        # Per-position chunk size s_k
        size_per_pos = torch.gather(
            chunk_out.chunk_sizes.squeeze(-1), dim=1, index=chunk_out.chunk_idx
        )
        s_2d = size_per_pos.index_select(1, model.raster_to_hilbert).view(B, G, G)

        all_detail.append(local_detail_2d.cpu().flatten())
        all_prob.append(p_2d.cpu().flatten())
        all_bmask.append(b_2d.cpu().flatten())
        all_csize.append(s_2d.cpu().flatten())

    detail_vec = torch.cat(all_detail).numpy()
    prob_vec = torch.cat(all_prob).numpy()
    bmask_vec = torch.cat(all_bmask).numpy()
    csize_vec = torch.cat(all_csize).numpy()

    qs = np.quantile(detail_vec, [0.0, 0.25, 0.50, 0.75, 1.0])
    quartile_stats = {}
    labels = [
        "Q1_Flat_Background",
        "Q2_Smooth_Shading",
        "Q3_Moderate_Texture",
        "Q4_Sharp_Edges_Detail",
    ]
    print("\n=== Adaptive Compute Allocation by Local 2D Detail Quartile ===")
    print(f"{'Quartile':<24} | {'Mean p_boundary':<16} | {'Boundary Rate':<15} | {'Effective Comp':<15} | {'Mean Chunk Size':<15}")
    print("-" * 95)
    for i, name in enumerate(labels):
        if i < 3:
            mask = (detail_vec >= qs[i]) & (detail_vec < qs[i + 1])
        else:
            mask = detail_vec >= qs[i]
        mp = float(prob_vec[mask].mean())
        br = float(bmask_vec[mask].mean())
        eff_comp = 1.0 / max(br, 1e-6)
        ms = float(csize_vec[mask].mean())
        quartile_stats[name] = {
            "mean_p_boundary": mp,
            "boundary_rate": br,
            "effective_compression": eff_comp,
            "mean_chunk_size_patches": ms,
        }
        print(f"{name:<24} | {mp:<16.4f} | {br*100:<14.2f}% | {eff_comp:<14.2f}x | {ms:<14.2f} patches")

    return quartile_stats


@torch.no_grad()
def plot_validation_chunk_panel(
    model: HNetDiT2D,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    out_path: str,
    device: torch.device,
    t_val: float = 0.75,
):
    G = model.config.grid_size
    patch_size = model.config.patch_size
    n_cols = 8

    # Pick 8 visually diverse validation images
    indices = [0, 7, 15, 23, 31, 42, 55, 68]
    x_1 = val_images[indices].to(device).float() / 127.5 - 1.0
    y = val_labels[indices].to(device)
    t = torch.full((n_cols,), t_val, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(2026)
    noise = torch.randn(x_1.shape, device=device, generator=gen)
    x_t = t_val * x_1 + (1.0 - t_val) * noise

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        x1_pred, router_out, chunk_out = model(x_t, t, y)

    b_2d, c_2d = model.hilbert_mask_to_2d(router_out.boundary_mask, chunk_out.chunk_idx)
    p_2d = router_out.boundary_prob[..., 1].float().index_select(1, model.raster_to_hilbert).view(n_cols, G, G)
    size_per_pos = torch.gather(chunk_out.chunk_sizes.squeeze(-1), dim=1, index=chunk_out.chunk_idx)
    s_2d = size_per_pos.index_select(1, model.raster_to_hilbert).view(n_cols, G, G)

    fig, axes = plt.subplots(5, n_cols, figsize=(2.4 * n_cols, 12.2))
    row_titles = [
        "Ground Truth x_1",
        f"Predicted x_1 (t={t_val:.2f})",
        "Router Boundary Prob p_t",
        "Dynamic 2D Chunk Borders",
        "2D Hilbert Superpixels",
    ]

    for col in range(n_cols):
        gt_np = ((x_1[col].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        pred_np = ((x1_pred[col].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        p_np = p_2d[col].cpu().numpy()
        cid_np = c_2d[col].cpu().numpy()
        n_chunks = int(router_out.boundary_mask[col].sum().item())
        comp = model.config.seq_len / max(n_chunks, 1)

        contour_img, seg_img = render_chunk_contours_on_image(gt_np, cid_np, patch_size)

        axes[0, col].imshow(gt_np)
        axes[0, col].set_title(f"Class {y[col].item()}", fontsize=9, fontweight="bold")
        axes[1, col].imshow(pred_np)
        axes[2, col].imshow(p_np, cmap="magma", vmin=0.1, vmax=0.8)
        axes[3, col].imshow(contour_img)
        axes[3, col].set_title(f"{n_chunks} chunks ({comp:.2f}x)", fontsize=8.5)
        axes[4, col].imshow(seg_img)

        for row in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(row_titles[row], fontsize=10, fontweight="bold")

    plt.suptitle(
        "Bidirectional 2D H-Net-DiT on ImageNet-64: Emergent 2D Dynamic Chunking & x-Prediction",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[Plot] Saved validation chunk analysis figure to {out_path}")


@torch.no_grad()
def plot_ode_trajectory_chunks(
    model: HNetDiT2D,
    out_path: str,
    device: torch.device,
):
    patch_size = model.config.patch_size
    classes = torch.tensor([207, 980, 817, 323], device=device)  # Golden Retriever, Volcano, Sports Car, Monarch
    class_names = ["Golden Retriever (207)", "Volcano (980)", "Sports Car (817)", "Monarch Butterfly (323)"]

    _, snapshots = model.sample(
        batch_size=4,
        y=classes,
        steps=40,
        cfg_scale=2.0,
        device=device,
        seed=42,
        return_trajectories=True,
    )

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(4, n_snaps * 2, figsize=(2.2 * n_snaps * 2, 9.2))

    for row in range(4):
        for s_idx, snap in enumerate(snapshots):
            t_val = snap["t"]
            x1_np = ((snap["x1_pred"][row].permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
            cid_np = snap["chunk_id_2d"][row].numpy()
            n_chunks = int(snap["boundary_2d"][row].sum().item())
            comp = model.config.seq_len / max(n_chunks, 1)
            contour_img, _ = render_chunk_contours_on_image(x1_np, cid_np, patch_size)

            c0 = s_idx * 2
            c1 = s_idx * 2 + 1
            axes[row, c0].imshow(x1_np)
            axes[row, c1].imshow(contour_img)
            if row == 0:
                axes[row, c0].set_title(f"t={t_val:.2f} x1_pred", fontsize=8.5, fontweight="bold")
                axes[row, c1].set_title(f"t={t_val:.2f} ({comp:.1f}x)", fontsize=8.5, fontweight="bold")
            if s_idx == 0:
                axes[row, c0].set_ylabel(class_names[row], fontsize=9, fontweight="bold")
            axes[row, c0].set_xticks([])
            axes[row, c0].set_yticks([])
            axes[row, c1].set_xticks([])
            axes[row, c1].set_yticks([])

    plt.suptitle(
        "Rectified Flow ODE Trajectory (t=0 -> t=1): Evolution of Clean x_1 Prediction & 2D Dynamic Chunks",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[Plot] Saved ODE trajectory chunk figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-cache", type=str, default="/dev/shm/imagenet64_cache/val_4096.pt")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    val_data = torch.load(args.val_cache, map_location="cpu", weights_only=True)
    val_images, val_labels = val_data["images"], val_data["labels"]

    os.makedirs(args.out_dir, exist_ok=True)
    stats = analyze_entropy_vs_chunking(model, val_images, val_labels, device, t_val=0.75)
    with open(os.path.join(args.out_dir, "chunk_quartile_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    plot_validation_chunk_panel(
        model, val_images, val_labels, os.path.join(args.out_dir, "dit2d_val_chunk_analysis.png"), device
    )
    plot_ode_trajectory_chunks(
        model, os.path.join(args.out_dir, "dit2d_ode_trajectory_chunks.png"), device
    )


if __name__ == "__main__":
    main()
