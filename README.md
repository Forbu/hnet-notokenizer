# `hnet-notokenizer` — Byte-Level Hierarchical Network (H-Net)

End-to-end PyTorch implementation of **Hierarchical Networks (H-Net)** with **Dynamic Chunking (DC)** operating directly on raw UTF-8 bytes (`0..255`), inspired by Cartesia Research's *"From Tokens to Bytes"* (Parts I & II) and `goombalab/hnet`.

## Architecture Overview

Each hierarchy stage $s$ processes a sequence via 5 modules:
1. **Encoder ($\mathcal{E}^s$):** Lightweight causal SSM/Gated-Conv stack (`CausalSSMBlock`) operating at fine resolution $L_s$.
2. **Dynamic Chunker (`RoutingModule` + `ChunkLayer`):**
   - Computes cosine dissimilarity between adjacent states:
     $$p_t = \frac{1 - \cos(W_q \hat{x}_{t-1}, W_k \hat{x}_t)}{2}, \quad b_t = \mathbb{I}[p_t > 0.5]$$
   - Packs only the boundary vectors ($b_t = 1$) into a compressed sequence of length $M \approx L_s / N_s$.
3. **Main Network ($\mathcal{M}$):**
   - In a **1-stage H-Net**, this is the core **Causal Transformer** (RoPE + FlashAttention SDPA + SwiGLU).
   - In a **2-stage H-Net**, Stage 0's `main_network` is **recursively another `HNetStage` (Stage 1)** which compresses chunks a second time before running the core Causal Transformer.
4. **Dechunker (`DeChunkLayer` + STE Residual):**
   - **Smoothing Module:** Parallel associative EMA scan across chunks weighted by boundary confidence $p_k$:
     $$\tilde{z}_k = p_k z_k + (1 - p_k)\tilde{z}_{k-1}$$
   - **Causal Broadcast:** Repeats each smoothed chunk $\tilde{z}_k$ across all byte positions belonging to that chunk (`cumsum(b) - 1`).
   - **Straight-Through Estimator (STE) + Residual:** Combines dechunked output with the fp32 zero-initialized Encoder residual.
5. **Decoder ($\mathcal{D}^s$):** Lightweight causal SSM/Gated-Conv stack at fine resolution $L_s$.

### Autoregressive Generation (`step` mode)
During generation (`model.generate()`), the model generates **1 byte at a time**:
- The lightweight outer **Encoder** and **Decoder** step on every byte.
- Whenever `boundary_mask == False` ($b_t = 0$), the heavy **Main Network is completely skipped (0 FLOPs)** and the **Dechunker** simply reuses its cached chunk vector (`state.last_value`).

---

## Quickstart

### 1. Install
```bash
pip install -e .
```

### 2. Run Unit & Gradient Verification Tests
```bash
python tests/test_hnet.py
```

### 3. Local Smoke Test (Overfit a tiny text file in 30 seconds)
```bash
python train.py \
  --preset small_1stage \
  --local-file tests/sample_corpus.txt \
  --seq-len 256 \
  --batch-size 4 \
  --grad-accum 1 \
  --max-steps 100 \
  --eval-every 50 \
  --prompt "In computer science "
```

---

## H100 Experiments

### Experiment A: Fast Coherent English & Dynamic Chunking (~15 mins on 1x H100)
`TinyStories` has clean, simple vocabulary and grammar—ideal for verifying that the model learns word/phrase boundaries and generates proper sentences quickly:
```bash
python train.py \
  --preset medium_2stage_h100 \
  --dataset tinystories \
  --seq-len 2048 \
  --batch-size 32 \
  --grad-accum 2 \
  --max-steps 3000 \
  --eval-every 250 \
  --prompt "Once upon a time, there was a "
```

### Experiment B: FineWeb-Edu 10BT Scaling Run (1x H100)
Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`) directly in raw UTF-8 bytes using `bf16` + FlashAttention:
```bash
python train.py \
  --preset medium_2stage_h100 \
  --dataset fineweb-edu \
  --seq-len 2048 \
  --batch-size 32 \
  --grad-accum 4 \
  --max-steps 10000 \
  --eval-every 500 \
  --prompt "In computer science and artificial intelligence, "
```

### Generate & Inspect Learned Dynamic Chunks
```bash
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "Once upon a time, " \
  --max-new-bytes 256
```
Output prints both the decoded UTF-8 text and the Stage-0 dynamic chunk segmentation (`Once |upon |a |time, |there |was...`) along with the percentage of byte steps where the inner Transformer was skipped.
