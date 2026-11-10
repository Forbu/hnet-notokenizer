"""
Train & Evaluate Bidirectional 2D H-Net-DiT (Design C) with Rectified Flow (x-prediction)
on ImageNet-64 (`Forbu14/imagenet-64`).

Features:
  - Fast C++ PyArrow + TorchVision parallel PNG decoder into RAM (`uint8` (N, 3, 64, 64) tensors)
    for zero-bottleneck GPU training (~0.1ms/batch data loading).
  - Supports both learned dynamic 2D routing (`--routing dynamic`) and static 2x2 block
    baseline (`--routing uniform`) at identical parameter count & compression ratio N=4.0.
  - Per-stage learning rate multipliers (`2.0x` on outer Stage-0 2D-SSM/Router, `1.0x` on inner Main DiT).
  - Exponential Moving Average (EMA) model evaluation & ODE sampling (Euler x-prediction + CFG).
  - Automatic 2D Dynamic Chunk visualization (chunk boundary contours + superpixel maps on both
    validation reconstructions and class-conditional ODE generated samples).
"""

import argparse
import copy
import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as tvio

from hnet.dit2d import HNetDiT2D, HNetDiT2DConfig


import io
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFile, PngImagePlugin

ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


def _decode_parquet_shard_worker(args: Tuple[str, str, Optional[int]]) -> str:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
    pf, shard_cache_path, max_rows = args
    if os.path.exists(shard_cache_path):
        return shard_cache_path
    table = pq.read_table(pf, columns=["image", "label"])
    if max_rows is not None and table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    bytes_col = table["image"].combine_chunks().field("bytes")
    lbl_np = table["label"].to_numpy(zero_copy_only=False).astype(np.int64)
    n = len(lbl_np)
    out = np.empty((n, 3, 64, 64), dtype=np.uint8)
    out_lbls = np.empty((n,), dtype=np.int64)
    valid_idx = 0
    for i in range(n):
        try:
            b = bytes_col[i].as_py()
            im = Image.open(io.BytesIO(b)).convert("RGB")
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)  # (64, 64, 3)
            out[valid_idx] = arr.transpose(2, 0, 1)
            out_lbls[valid_idx] = lbl_np[i]
            valid_idx += 1
        except Exception:
            continue
    out = out[:valid_idx]
    out_lbls = out_lbls[:valid_idx]
    os.makedirs(os.path.dirname(shard_cache_path), exist_ok=True)
    tmp_path = f"{shard_cache_path}.tmp.{os.getpid()}"
    torch.save({"images": torch.from_numpy(out), "labels": torch.from_numpy(out_lbls)}, tmp_path)
    os.replace(tmp_path, shard_cache_path)
    return shard_cache_path


def load_parquet_shards_to_ram(
    parquet_files: List[str],
    cache_path: Optional[str] = None,
    max_images: Optional[int] = None,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode ImageNet-64 parquet shards in parallel across CPU processes into a contiguous
    (N, 3, 64, 64) uint8 tensor and (N,) int64 labels. Saves per-shard caches in /dev/shm for instant loads.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Data] Loading cached uint8 tensor from {cache_path} ...", flush=True)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        imgs, labels = data["images"], data["labels"]
        if max_images is not None and imgs.shape[0] > max_images:
            imgs = imgs[:max_images]
            labels = labels[:max_images]
        print(f"[Data] Loaded {imgs.shape[0]:,} images ({imgs.numel() / 1e9:.2f} GB uint8)", flush=True)
        return imgs, labels

    t0 = time.time()
    cache_dir = os.path.dirname(cache_path) if cache_path else "/dev/shm/imagenet64_cache"
    os.makedirs(cache_dir, exist_ok=True)

    tasks = []
    for pf in parquet_files:
        base = os.path.basename(pf).replace(".parquet", "")
        max_r = max_images if len(parquet_files) == 1 else None
        sc_path = os.path.join(cache_dir, f"shard_{base}_{max_r or 'all'}.pt")
        tasks.append((pf, sc_path, max_r))

    workers = min(num_workers, len(tasks), os.cpu_count() or 8)
    print(f"[Data] Decoding {len(tasks)} parquet shard(s) across {workers} CPU processes ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        shard_paths = list(ex.map(_decode_parquet_shard_worker, tasks))

    all_imgs = []
    all_labels = []
    for sp in shard_paths:
        d = torch.load(sp, map_location="cpu", weights_only=True)
        all_imgs.append(d["images"])
        all_labels.append(d["labels"])

    images = torch.cat(all_imgs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if max_images is not None and images.shape[0] > max_images:
        images = images[:max_images]
        labels = labels[:max_images]

    print(
        f"[Data] Decoded & loaded {images.shape[0]:,} images ({images.numel() / 1e9:.2f} GB uint8) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return images, labels


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def build_optimizer(model: HNetDiT2D, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    groups: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        stage_idx = getattr(p, "_stage_idx", 0)
        no_decay = (
            p.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "adaLN" in name
            or "bos_embed" in name
            or "pad_dimension" in name
            or "residual_proj" in name
            or "class_embed" in name
        )
        key = (stage_idx, no_decay)
        groups.setdefault(key, []).append(p)

    param_groups = []
    for (stage_idx, no_decay), params in sorted(groups.items()):
        mult = model.config.lr_multipliers[min(stage_idx, len(model.config.lr_multipliers) - 1)]
        param_groups.append(
            {
                "params": params,
                "lr": lr * mult,
                "lr_mult": mult,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


def cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(warmup_steps, 1))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def render_chunk_contours_on_image(
    img_01: np.ndarray, chunk_id_2d: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a clean or predicted RGB image in [0, 1] of shape (H, W, 3) and `chunk_id_2d` of shape (G, G):
      1. Upsamples `chunk_id_2d` to (H, W) by nearest-neighbor (exact patch blocks)
      2. Draws crisp red boundaries wherever adjacent pixels belong to different dynamic chunks
      3. Creates a distinct pastel color map of the chunks blended with the image
    """
    H, W, _ = img_01.shape
    chunk_pixel = np.repeat(np.repeat(chunk_id_2d, patch_size, axis=0), patch_size, axis=1)

    # Boundary mask where right or bottom neighbor has a different chunk ID
    edge = np.zeros((H, W), dtype=bool)
    edge[:-1, :] |= chunk_pixel[:-1, :] != chunk_pixel[1:, :]
    edge[:, :-1] |= chunk_pixel[:, :-1] != chunk_pixel[:, 1:]

    overlay = img_01.copy()
    # Bright red/coral contour lines on chunk borders
    overlay[edge] = np.array([1.0, 0.18, 0.18])

    # Golden-ratio color palette for chunk IDs
    phi = 0.618033988749895
    hues = (chunk_pixel * phi) % 1.0
    cmap = plt.get_cmap("tab20")(hues)
    seg_rgb = 0.55 * img_01 + 0.45 * cmap[..., :3]
    seg_rgb[edge] = np.array([0.05, 0.05, 0.05])
    return overlay, seg_rgb


@torch.no_grad()
def evaluate_validation(
    model: HNetDiT2D,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    Deterministic validation across a fixed grid of flow timesteps t in {0.2, 0.4, 0.6, 0.8, 0.9}.
    """
    model.eval()
    eval_ts = [0.2, 0.4, 0.6, 0.8, 0.9]
    gen = torch.Generator(device=device).manual_seed(12345)

    total_flow_loss = 0.0
    total_x1_mse = 0.0
    total_comp = 0.0
    count = 0

    n_eval = min(val_images.shape[0], 1024)
    for t_val in eval_ts:
        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            B = end - start
            x_1 = val_images[start:end].to(device, non_blocking=True).float() / 127.5 - 1.0
            y = val_labels[start:end].to(device, non_blocking=True)
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            noise = torch.randn(x_1.shape, device=device, generator=gen)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, metrics = model.forward_loss(x_1, y=y, t=t, noise=noise)

            total_flow_loss += metrics["flow_loss"] * B
            total_x1_mse += metrics["x1_mse"] * B
            total_comp += metrics["compression_ratio"] * B
            count += B

    return {
        "val_flow_loss": total_flow_loss / max(count, 1),
        "val_x1_mse": total_x1_mse / max(count, 1),
        "val_compression": total_comp / max(count, 1),
    }


@torch.no_grad()
def save_visualizations(
    model: HNetDiT2D,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    save_dir: str,
    step: int,
    device: torch.device,
):
    """
    Save two diagnostic visualization figures:
      1. `val_chunks_step_{step:06d}.png`: Real validation images at t=0.70 showing
         Ground Truth x_1 | Noisy x_t | Predicted x_1 | Dynamic 2D Chunk Contours | Superpixel Map
      2. `ode_samples_step_{step:06d}.png`: Class-conditional Rectified Flow ODE samples (50 steps, CFG=2.0)
         alongside their learned 2D dynamic chunk segmentation maps.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    patch_size = model.config.patch_size

    # --- Figure 1: Validation Denoising & Emergent 2D Dynamic Chunks ---
    n_vis = 6
    x_1 = val_images[:n_vis].to(device).float() / 127.5 - 1.0
    y = val_labels[:n_vis].to(device)
    t = torch.full((n_vis,), 0.70, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(999)
    noise = torch.randn(x_1.shape, device=device, generator=gen)
    x_t = 0.70 * x_1 + 0.30 * noise

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        x1_pred, router_out, chunk_out = model(x_t, t, y)

    b_2d, c_2d = model.hilbert_mask_to_2d(router_out.boundary_mask, chunk_out.chunk_idx)

    fig, axes = plt.subplots(n_vis, 5, figsize=(15, 3.0 * n_vis))
    for i in range(n_vis):
        gt_np = ((x_1[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        xt_np = ((x_t[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        pred_np = ((x1_pred[i].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        cid_np = c_2d[i].cpu().numpy()
        n_chunks_i = int(router_out.boundary_mask[i].sum().item())
        comp_i = model.config.seq_len / max(n_chunks_i, 1)

        contour_img, seg_img = render_chunk_contours_on_image(pred_np, cid_np, patch_size)

        axes[i, 0].imshow(gt_np)
        axes[i, 0].set_title(f"GT x_1 (class {y[i].item()})", fontsize=9)
        axes[i, 1].imshow(xt_np)
        axes[i, 1].set_title("Noisy Input x_t (t=0.70)", fontsize=9)
        axes[i, 2].imshow(pred_np)
        axes[i, 2].set_title("Predicted Clean x_1", fontsize=9)
        axes[i, 3].imshow(contour_img)
        axes[i, 3].set_title(f"2D Chunk Borders ({n_chunks_i} chunks, {comp_i:.2f}x)", fontsize=9)
        axes[i, 4].imshow(seg_img)
        axes[i, 4].set_title("2D Hilbert Superpixels", fontsize=9)
        for j in range(5):
            axes[i, j].axis("off")

    plt.suptitle(
        f"2D H-Net-DiT ({model.config.routing_mode}) @ Step {step:,} — Validation x-Prediction & Dynamic 2D Chunks",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    val_fig_path = os.path.join(save_dir, f"val_chunks_step_{step:06d}.png")
    plt.savefig(val_fig_path, dpi=140)
    plt.close(fig)

    # --- Figure 2: ODE Generated Samples + Learned 2D Chunk Maps ---
    # Fixed diverse ImageNet classes: e.g. golden retriever, volcano, sports car, monarch butterfly, lorikeet, castle, speedboat, daisy
    sample_classes = torch.tensor([207, 980, 817, 323, 90, 483, 814, 985], device=device)
    samples, traj = model.sample(
        batch_size=8,
        y=sample_classes,
        steps=40,
        cfg_scale=2.0,
        device=device,
        seed=42,
        return_trajectories=True,
    )
    final_snap = traj[-1]

    fig2, axes2 = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(8):
        row = i // 2
        col_base = (i % 2) * 2
        samp_np = ((samples[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0, 1)
        cid_np = final_snap["chunk_id_2d"][i].numpy()
        n_chunks_i = int(final_snap["boundary_2d"][i].sum().item())
        comp_i = model.config.seq_len / max(n_chunks_i, 1)
        contour_img, _ = render_chunk_contours_on_image(samp_np, cid_np, patch_size)

        axes2[row, col_base].imshow(samp_np)
        axes2[row, col_base].set_title(f"ODE Sample (cls {sample_classes[i].item()})", fontsize=9)
        axes2[row, col_base].axis("off")

        axes2[row, col_base + 1].imshow(contour_img)
        axes2[row, col_base + 1].set_title(
            f"Dynamic Chunks ({n_chunks_i} / {comp_i:.2f}x)", fontsize=9
        )
        axes2[row, col_base + 1].axis("off")

    plt.suptitle(
        f"2D H-Net-DiT ({model.config.routing_mode}) @ Step {step:,} — Rectified Flow ODE Samples & 2D Chunks",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    ode_fig_path = os.path.join(save_dir, f"ode_samples_step_{step:06d}.png")
    plt.savefig(ode_fig_path, dpi=140)
    plt.close(fig2)
    print(f"[Vis] Saved {val_fig_path} and {ode_fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Bidirectional 2D H-Net-DiT on ImageNet-64")
    parser.add_argument("--preset", type=str, default="dit2d_base", choices=["dit2d_small", "dit2d_base", "dit2d_large"])
    parser.add_argument("--patch-size", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--routing", type=str, default="dynamic", choices=["dynamic", "uniform"])
    parser.add_argument("--target-compression", type=float, default=4.0)
    parser.add_argument("--aux-loss-weight", type=float, default=0.20)
    parser.add_argument("--data-dir", type=str, default="/workspace/imagenet64_hf/data")
    parser.add_argument("--cache-dir", type=str, default="/dev/shm/imagenet64_cache")
    parser.add_argument("--max-train-shards", type=int, default=10, help="Number of train parquet shards to load into RAM")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--vis-every", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="checkpoints_dit2d")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop cleanly after N hours of training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.preset == "dit2d_small":
        cfg = HNetDiT2DConfig.dit2d_small(patch_size=args.patch_size, routing_mode=args.routing)
    elif args.preset == "dit2d_base":
        cfg = HNetDiT2DConfig.dit2d_base(patch_size=args.patch_size, routing_mode=args.routing)
    else:
        cfg = HNetDiT2DConfig.dit2d_large(patch_size=args.patch_size, routing_mode=args.routing)
    cfg.target_compression = args.target_compression
    cfg.aux_loss_weight = args.aux_loss_weight

    model = HNetDiT2D(cfg).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {args.preset} | routing={cfg.routing_mode} | patch={cfg.patch_size}x{cfg.patch_size} "
        f"(grid {cfg.grid_size}x{cfg.grid_size}, L={cfg.seq_len} -> M~{int(cfg.seq_len/cfg.target_compression)}) "
        f"| Params: {n_params / 1e6:.2f}M"
    )

    # Load validation and training shards into RAM
    val_files = sorted(glob.glob(os.path.join(args.data_dir, "validation-*.parquet")))
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))[: args.max_train_shards]
    assert len(val_files) > 0 and len(train_files) > 0, f"No parquet files found in {args.data_dir}"

    val_cache = os.path.join(args.cache_dir, "val_4096.pt") if args.cache_dir else None
    train_tag = f"train_s{len(train_files)}_{args.max_train_images or 'all'}.pt"
    train_cache = os.path.join(args.cache_dir, train_tag) if args.cache_dir else None

    val_images, val_labels = load_parquet_shards_to_ram(val_files, cache_path=val_cache, max_images=4096)
    train_images, train_labels = load_parquet_shards_to_ram(
        train_files, cache_path=train_cache, max_images=args.max_train_images
    )
    num_train = train_images.shape[0]
    print(f"[Data] Ready: {num_train:,} train images, {val_images.shape[0]:,} val images in RAM.")

    os.makedirs(args.save_dir, exist_ok=True)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    ema_flow_loss = None
    best_val_loss = float("inf")
    history = []

    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading checkpoint from {args.resume} ...", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(ckpt.get("step", 0))
        if "raw_model" in ckpt:
            model.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["raw_model"].items()})
            ema_model.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["model"].items()})
        else:
            state = {k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["model"].items()}
            model.load_state_dict(state)
            ema_model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "val_metrics" in ckpt and "val_flow_loss" in ckpt["val_metrics"]:
            best_val_loss = float(ckpt["val_metrics"]["val_flow_loss"])
        metrics_file = os.path.join(args.save_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                history = json.load(f)
        print(f"[Resume] Resumed at step {start_step:,} (best_val_loss={best_val_loss:.4f})", flush=True)

    t_start = time.time()
    steps_run = 0

    if start_step == 0:
        val_metrics = evaluate_validation(ema_model, val_images, val_labels, device, batch_size=args.batch_size)
        print(
            f"[Eval Step 0] val_flow_loss: {val_metrics['val_flow_loss']:.4f} | "
            f"val_x1_mse: {val_metrics['val_x1_mse']:.4f} | val_comp: {val_metrics['val_compression']:.2f}x",
            flush=True,
        )

    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        cur_lr = cosine_lr(step - 1, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        set_optimizer_lr(optimizer, cur_lr)
        optimizer.zero_grad(set_to_none=True)

        step_flow_loss = 0.0
        step_x1_mse = 0.0
        step_r_loss = 0.0
        step_comp = 0.0
        step_bprob = 0.0

        for micro in range(args.grad_accum):
            idx = torch.randint(0, num_train, (args.batch_size,))
            x_1 = train_images[idx].to(device, non_blocking=True).float() / 127.5 - 1.0
            # Random horizontal flip augmentation
            flip_mask = torch.rand(args.batch_size, 1, 1, 1, device=device) < 0.5
            x_1 = torch.where(flip_mask, x_1.flip(-1), x_1)
            y = train_labels[idx].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, metrics = model.forward_loss(x_1, y)
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            step_flow_loss += metrics["flow_loss"] / args.grad_accum
            step_x1_mse += metrics["x1_mse"] / args.grad_accum
            step_r_loss += metrics["ratio_loss"] / args.grad_accum
            step_comp += metrics["compression_ratio"] / args.grad_accum
            step_bprob += metrics["boundary_prob"] / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        update_ema(ema_model, model, args.ema_decay)
        steps_run += 1

        if ema_flow_loss is None:
            ema_flow_loss = step_flow_loss
        else:
            ema_flow_loss = 0.98 * ema_flow_loss + 0.02 * step_flow_loss

        elapsed = time.time() - t_start
        time_limit_hit = (args.max_hours is not None) and (elapsed >= args.max_hours * 3600.0)
        is_final = (step == args.max_steps) or time_limit_hit

        if step % args.log_every == 0 or steps_run == 1 or is_final:
            imgs_per_sec = (steps_run * args.batch_size * args.grad_accum) / max(elapsed, 1e-3)
            hrs = elapsed / 3600.0
            print(
                f"Step {step:5d}/{args.max_steps} ({hrs:.2f}h) | flow_loss: {step_flow_loss:.4f} (EMA: {ema_flow_loss:.4f}) | "
                f"x1_mse: {step_x1_mse:.4f} | r_loss: {step_r_loss:.4f} | "
                f"comp: {step_comp:.2f}x (p={step_bprob:.3f}) | grad: {grad_norm:.2f} | "
                f"lr: {cur_lr:.2e} | {imgs_per_sec:.1f} img/s",
                flush=True,
            )

        if step % args.eval_every == 0 or is_final:
            val_m = evaluate_validation(ema_model, val_images, val_labels, device, batch_size=args.batch_size)
            raw_m = evaluate_validation(model, val_images, val_labels, device, batch_size=args.batch_size)
            print(
                f"=== [Eval Step {step:5d} ({elapsed/3600.0:.2f}h)] EMA val_flow_loss: {val_m['val_flow_loss']:.4f} "
                f"(raw: {raw_m['val_flow_loss']:.4f}) | EMA val_x1_mse: {val_m['val_x1_mse']:.4f} "
                f"| comp: {val_m['val_compression']:.2f}x ===",
                flush=True,
            )
            rec = {
                "step": step,
                "elapsed_hours": round(elapsed / 3600.0, 4),
                "train_flow_loss": step_flow_loss,
                "ema_train_flow_loss": ema_flow_loss,
                "val_flow_loss": val_m["val_flow_loss"],
                "raw_val_flow_loss": raw_m["val_flow_loss"],
                "val_x1_mse": val_m["val_x1_mse"],
                "val_compression": val_m["val_compression"],
                "train_compression": step_comp,
            }
            history.append(rec)
            with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
                json.dump(history, f, indent=2)

            latest_payload = {
                "step": step,
                "config": cfg.__dict__,
                "model": {k: v.to(torch.bfloat16).cpu() for k, v in ema_model.state_dict().items()},
                "raw_model": {k: v.to(torch.bfloat16).cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_m,
            }
            torch.save(latest_payload, os.path.join(args.save_dir, "latest.pt"))

            if val_m["val_flow_loss"] < best_val_loss:
                best_val_loss = val_m["val_flow_loss"]
                torch.save(
                    {
                        "step": step,
                        "config": cfg.__dict__,
                        "model": latest_payload["model"],
                        "val_metrics": val_m,
                    },
                    os.path.join(args.save_dir, "best_ema.pt"),
                )

            if step % 5000 == 0 or is_final:
                torch.save(
                    {
                        "step": step,
                        "config": cfg.__dict__,
                        "model": latest_payload["model"],
                        "val_metrics": val_m,
                    },
                    os.path.join(args.save_dir, f"step_{step:06d}_ema.pt"),
                )

        if step % args.vis_every == 0 or is_final:
            save_visualizations(ema_model, val_images, val_labels, args.save_dir, step, device)

        if time_limit_hit:
            print(f"[Done] Reached max_hours={args.max_hours:.2f}h at step {step:,}. Exiting cleanly.", flush=True)
            break


if __name__ == "__main__":
    main()
