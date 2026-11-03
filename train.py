#!/usr/bin/env python3
"""
Train a Byte-Level Hierarchical Network (H-Net) with Dynamic Chunking on FineWeb-Edu or TinyStories.

Example (Quick local smoke test):
    python train.py --preset small_1stage --local-file tests/sample_corpus.txt --max-steps 50 --seq-len 256 --batch-size 4

Example (H100 run on FineWeb-Edu 10BT):
    python train.py --preset medium_2stage_h100 --dataset fineweb-edu --seq-len 2048 --batch-size 32 --grad-accum 2 --max-steps 10000

Example (Fast H100 run on TinyStories to see coherent stories & dynamic chunks in ~15 mins):
    python train.py --preset medium_2stage_h100 --dataset tinystories --seq-len 2048 --batch-size 32 --max-steps 3000
"""

import argparse
import math
import os
import time

import torch

from hnet.config import HNetConfig
from hnet.data import build_byte_dataloader
from hnet.model import ByteHNetLM


def get_Config(preset: str) -> HNetConfig:
    if preset == "small_1stage":
        return HNetConfig.small_1stage()
    elif preset == "hnet_1stage_l12":
        return HNetConfig.hnet_1stage_l12()
    elif preset == "hnet_1stage_l16":
        return HNetConfig.hnet_1stage_l16()
    elif preset == "medium_2stage_h100":
        return HNetConfig.medium_2stage_h100()
    raise ValueError(f"Unknown preset: {preset}")


def get_lr_scale(step: int, max_steps: int, warmup_steps: int) -> float:
    """Warmup + Cosine Decay schedule."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main():
    parser = argparse.ArgumentParser(description="Train Byte-Level H-Net (No Tokenizer)")
    parser.add_argument(
        "--preset",
        type=str,
        default="hnet_1stage_l12",
        choices=["small_1stage", "hnet_1stage_l12", "hnet_1stage_l16", "medium_2stage_h100"],
    )
    parser.add_argument("--dataset", type=str, default="fineweb-edu", help="fineweb-edu or tinystories")
    parser.add_argument("--local-file", type=str, default=None, help="Optional local UTF-8 text file for offline test")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length in bytes")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size per step")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=5000, help="Total optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate (scaled per stage)")
    parser.add_argument("--aux-weight", type=float, default=0.10, help="Auxiliary compression ratio loss weight")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=200, help="Print sample generation + chunk boundaries every N steps")
    parser.add_argument("--prompt", type=str, default="In computer science and artificial intelligence, ", help="Prompt for live sample generation")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device & precision setup (auto-enables bf16 + TF32 on H100/Ampere+)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False
        amp_dtype = torch.float32
    else:
        device = torch.device("cpu")
        use_amp = False
        amp_dtype = torch.float32

    config = get_Config(args.preset)
    config.aux_loss_weight = args.aux_weight
    model = ByteHNetLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print(f"ByteHNetLM ({args.preset}) | Stages: {config.num_stages} | Total Params: {total_params / 1e6:.2f}M")
    print(f"Dimensions: {config.d_model} | Target Compression: {config.target_compression}")
    print(f"Device: {device} | Mixed Precision: {use_amp} ({amp_dtype})")
    print("=" * 80)

    param_groups = model.get_param_groups(base_lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    start_step = 1
    best_bpb = float("inf")
    skip_docs = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0)) + 1
        best_bpb = float(ckpt.get("bpb", float("inf")))
        # Estimate ~30 docs per step (131KB / ~4KB per FineWeb doc) so we stream fresh documents
        skip_docs = (start_step - 1) * 30
        print(f"Resumed from {args.resume} at step {start_step - 1} (BPB: {best_bpb:.4f}, skipping ~{skip_docs} docs)")

    train_loader = build_byte_dataloader(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        local_file=args.local_file,
        skip_docs=skip_docs,
    )
    train_iter = iter(train_loader)

    t0 = time.time()
    bytes_processed = 0

    for step in range(start_step, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Update learning rate with per-stage multipliers
        lr_scale = get_lr_scale(step - 1, args.max_steps, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * pg["lr_mult"] * lr_scale

        accum_loss = 0.0
        accum_bpb = 0.0
        accum_aux = 0.0
        comp_ratios = [0.0] * config.num_stages

        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=torch.cuda.is_available())
            y = y.to(device, non_blocking=torch.cuda.is_available())
            bytes_processed += x.numel()

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(x, targets=y)
                    loss = out["loss"] / args.grad_accum
            else:
                out = model(x, targets=y)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            accum_loss += out["ce_loss"].item() / args.grad_accum
            accum_bpb += out["bpb"].item() / args.grad_accum
            accum_aux += out["aux_loss"].item() / args.grad_accum
            for i, r in enumerate(out["compression_ratios"]):
                comp_ratios[i] += r / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            dt = time.time() - t0
            bps = bytes_processed / max(dt, 1e-5)
            comp_str = " * ".join(f"S{i}:{r:.2f}x" for i, r in enumerate(comp_ratios))
            total_comp = math.prod(comp_ratios)
            print(
                f"Step {step:5d}/{args.max_steps} | BPB: {accum_bpb:.4f} | CE: {accum_loss:.4f} | "
                f"Aux: {accum_aux:.4f} | Chunks: [{comp_str} => {total_comp:.2f}B/chunk] | "
                f"Speed: {bps / 1e3:.1f} KB/s"
            )
            t0 = time.time()
            bytes_processed = 0

        if step % args.eval_every == 0 or step == args.max_steps:
            text, chunk_view, gen_stats = model.generate(
                prompt=args.prompt, max_new_bytes=120, temperature=0.8, top_k=40
            )
            print("-" * 80)
            print(f"[Sample @ Step {step}] (Core Compression: {gen_stats['bytes_per_chunk']:.2f} B/chunk | "
                  f"Core Transformer Skipped: {gen_stats['main_net_skip_pct']:.1f}% of bytes)")
            print(f"Text   : {text!r}")
            print(f"Chunks :\n{chunk_view}")
            print("-" * 80)

            ckpt = {
                "step": step,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "bpb": accum_bpb,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "latest.pt"))
            if accum_bpb < best_bpb:
                best_bpb = accum_bpb
                torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))


    del train_iter, train_loader


if __name__ == "__main__":
    main()
