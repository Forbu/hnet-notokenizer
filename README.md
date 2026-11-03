# `hnet-notokenizer` — Byte-Level Hierarchical Network (H-Net)

End-to-end PyTorch + Triton implementation of **Hierarchical Networks (H-Net)** with **Dynamic Chunking (DC)** operating directly on raw UTF-8 bytes (`0..255`), based on Cartesia Research's *"From Tokens to Bytes"* (Parts I & II) and `goombalab/hnet`.

## Key Architectural Details

Each hierarchy stage $s$ processes a sequence via 5 modules:
1. **Encoder ($\mathcal{E}^s$, layout `"m4"`):** 4 pure Mamba-style causal SSM blocks (`CausalSSMBlock`, depthwise `Conv1d(k=4)` + parallel gated linear recurrence, no FFN) operating at byte resolution $L$.
2. **Dynamic Chunker (`RoutingModule` + `ChunkLayer`):**
   - Predicts chunk boundaries via cosine dissimilarity of adjacent states:
     $$p_t = \frac{1 - \cos(W_q \hat{x}_{t-1}, W_k \hat{x}_t)}{2}, \quad b_t = \mathbb{I}[p_t > 0.5]$$
   - Packs only the boundary vectors ($b_t = 1$) into a compressed sequence of length $M \approx L / N$ (default $N = 4.57$, matching BPE compression).
3. **Main Network ($\mathcal{M}$, layout `"T12"` / `"T16"`):**
   - Standard **Causal Transformer** operating on the $4.57\times$ compressed chunks:
     - **50% Partial RoPE:** Rotary Position Embeddings applied to the first half of each head's channels (`rotary_emb_dim = head_dim // 2`, e.g., `64` of `128`), leaving the remaining half as position-invariant content channels.
     - **FlashAttention SDPA** (`window_size = -1` for core Transformer, `window_size = 1023` for outer hybrid stages).
     - **SwiGLU MLP** with intermediate dimension $3.6 \times D$ rounded to a multiple of 128.
     - **Cumulative Depth-Scaled Residual Initialization:** `out_proj` and `fc2` initialized with $\text{std} = 0.02 / \sqrt{N_{\text{residuals}}}$.
4. **Dechunker (`DeChunkLayer` + STE Residual):**
   - **Fused Triton EMA Scan (`_TritonEMAScan`):** Smooths chunk representations weighted by boundary confidence $p_k$:
     $$\tilde{z}_k = p_k z_k + (1 - p_k)\tilde{z}_{k-1}$$
   - **Causal Broadcast:** Repeats each smoothed chunk $\tilde{z}_k$ across all byte positions belonging to that chunk (`cumsum(b) - 1`).
   - **Straight-Through Estimator (STE) + Residual:** Combines dechunked output with the fp32 zero-initialized Encoder residual.
5. **Decoder ($\mathcal{D}^s$, layout `"m4"`):** 4 pure causal SSM blocks at byte resolution $L$, followed by the byte LM head (`Linear(D, 256)`).

### Autoregressive Generation (`step` mode)
During generation (`model.generate()`), the model generates **1 byte at a time**:
- The lightweight outer **Encoder** and **Decoder** step on every byte.
- Whenever `boundary_mask == False` ($b_t = 0$, ~78% of byte steps), the heavy **Main Transformer is completely skipped (0 FLOPs)** and the **Dechunker** reuses its cached chunk vector (`state.last_value`).

---

## Quickstart

### 1. Install & Verify
```bash
pip install -e .
python tests/test_hnet.py
```

### 2. Train 1-Stage Byte H-Net on FineWeb-Edu (`151M` params)
```bash
python train.py \
  --preset hnet_1stage_l12 \
  --dataset fineweb-edu \
  --seq-len 2048 \
  --batch-size 64 \
  --grad-accum 1 \
  --max-steps 5000 \
  --warmup-steps 100 \
  --eval-every 250 \
  --save-dir checkpoints_fineweb_1stage \
  --prompt "In computer science and artificial intelligence, "
```
To resume from a saved checkpoint:
```bash
python train.py \
  --preset hnet_1stage_l12 \
  --dataset fineweb-edu \
  --seq-len 2048 \
  --batch-size 64 \
  --max-steps 5000 \
  --save-dir checkpoints_fineweb_1stage \
  --resume checkpoints_fineweb_1stage/latest.pt
```

### 3. Generate Text & Inspect Dynamic Chunk Boundaries
```bash
python generate.py \
  --checkpoint checkpoints_fineweb_1stage/best.pt \
  --prompt "The history of the Roman Empire began when " \
  --max-new-bytes 220 \
  --temperature 0.75
```
