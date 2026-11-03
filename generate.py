#!/usr/bin/env python3
"""
Generate text byte-by-byte and inspect dynamic H-Net chunk boundaries from a trained checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time, "
"""

import argparse
import torch
from hnet.model import ByteHNetLM


def main():
    parser = argparse.ArgumentParser(description="Generate bytes & visualize dynamic chunks with H-Net")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="The meaning of intelligence is ")
    parser.add_argument("--max-new-bytes", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = ByteHNetLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text, chunk_view, stats = model.generate(
        prompt=args.prompt,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("=" * 80)
    print(f"Checkpoint Step : {ckpt.get('step', '?')} (Train BPB: {ckpt.get('bpb', 0.0):.4f})")
    print(f"Compression     : {stats['bytes_per_chunk']:.2f} bytes/chunk "
          f"(Main Network skipped on {stats['main_net_skip_pct']:.1f}% of byte steps)")
    print("=" * 80)
    print("\n[Generated Text]")
    print(text)
    print("\n[Stage-0 Dynamic Chunks ('|' = learned boundary)]")
    print(chunk_view)
    print("=" * 80)


if __name__ == "__main__":
    main()
