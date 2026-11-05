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
  --checkpoint checkpoints/step_110000_ema1.05.pt \
  --prompt "During the Industrial Revolution, the invention of the steam engine " \
  --max-new-bytes 280 \
  --temperature 0.65
```

---

## 13-Hour FineWeb-Edu Training Run (`110,000` Steps / `14.42 GB` Raw UTF-8 Bytes)

We trained **`hnet_1stage_l12`** (**151.24M parameters**, layout `["m4", ["T12"], "m4"]`, $D = [512, 896]$, target compression $N = 4.57$) from scratch on **FineWeb-Edu (`sample-10BT`)** directly on raw UTF-8 bytes without any tokenizer.

### Training Setup
* **Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (`96 GB` VRAM, `bfloat16` mixed precision + fused Triton EMA scan).
* **Batch Size:** `64 × 2048 = 131,072 bytes/step` (`~265 KB/s` steady throughput).
* **Total Horizon:** **110,000 steps** (~13 hours), processing **14.42 GB of raw UTF-8 text** (**~3.15 billion BPE-token equivalents**, full Chinchilla compute-optimal scale for a 151M parameter model).
* **Logs & Local Weights:**
  * Full training logs: `logs/train_5k.log` and `logs/train_13h.log`
  * Exported `bfloat16` inference checkpoints (`289 MB` each):
    * `checkpoints/best_step102000_ema1.00.pt` (`Step 102,000`, `EMA BPB: 1.0076`)
    * `checkpoints/step_110000_ema1.05.pt` (`Step 110,000`, `EMA BPB: 1.0542`)

### 110,000-Step Scaling Curve

| Step | Raw Text Processed | Batch BPB | Smoothed EMA BPB | Cross-Entropy (nats) | Empirical Compression | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.13 MB | `8.0554` | `8.0554` | `5.5836` | `2.22 B/chunk` | `3.00e-06` |
| **5,000** | 0.66 GB | `1.5866` | `1.5866` | `1.0997` | `4.43 B/chunk` | `3.00e-05` |
| **10,000** | 1.31 GB | `1.4813` | `1.4334` | `1.0267` | `3.50 B/chunk` | `2.95e-04` |
| **20,000** | 2.62 GB | `1.1645` | `1.1800` | `0.8071` | `3.72 B/chunk` | `2.79e-04` |
| **30,000** | 3.93 GB | `1.1074` | `1.1751` | `0.7676` | `3.63 B/chunk` | `2.54e-04` |
| **40,000** | 5.24 GB | `1.1300` | `1.1434` | `0.7832` | `3.53 B/chunk` | `2.22e-04` |
| **50,000** | 6.55 GB | `1.0748` | `1.0972` | `0.7450` | `3.76 B/chunk` | `1.85e-04` |
| **60,000** | 7.86 GB | `1.0785` | `1.1069` | `0.7475` | `3.91 B/chunk` | `1.46e-04` |
| **70,000** | 9.18 GB | `1.0244` | `1.0871` | `0.7101` | `4.22 B/chunk` | `1.09e-04` |
| **80,000** | 10.49 GB | `1.0363` | `1.0685` | `0.7183` | `4.14 B/chunk` | `7.68e-05` |
| **90,000** | 11.80 GB | `1.0163` | `1.0877` | `0.7045` | `4.13 B/chunk` | `5.15e-05` |
| **100,000** | 13.11 GB | `0.9756` | `1.0430` | `0.6762` | `4.40 B/chunk` | `3.55e-05` |
| **106,760** | 13.99 GB | **`0.9118`** *(min)* | `1.0214` | `0.6320` | `4.33 B/chunk` | `3.06e-05` |
| **110,000** | **14.42 GB** | **`1.0088`** | **`1.0542`** | `0.6992` | **`4.16 B/chunk`** | `3.00e-05` |

#### Observed Two-Phase Router Learning Dynamics
1. **Subword Discovery Phase (Steps 10k–40k):** The cosine-dissimilarity router temporarily tightens empirical compression to `~3.5–3.7 bytes/chunk` while the outer Mamba-style encoder/decoder (`m4`) learns common morphological prefixes, suffixes, and stems.
2. **Whole-Word & Phrase Consolidation Phase (Steps 50k–110k):** Once word representations solidify in the outer SSM stacks, the router smoothly expands its chunks back up to **`4.16–4.45 bytes/chunk`** during training (**`4.5–6.1 bytes/chunk` during generation**), driving Bits-Per-Byte below `1.00` on individual batches (`0.9118 BPB` min) and `~1.00–1.05` EMA BPB.

---

### Generated Samples & Emergent Dynamic Chunking (`Step 110,000`)

In every sample below, `|` denotes a learned boundary where `boundary_mask == True` ($p_t > 0.5$). On all bytes between `|` markers (~78%–84% of byte steps), the 12-layer Main Transformer is **completely skipped during autoregressive generation**.

#### 1. Computer Science & AI (`5.50 bytes/chunk` — Main Transformer skipped on `81.8%` of bytes)
* **Generated Text:**
  > `In computer science and artificial intelligence, linear algebra is a computational framework that allows the software to determine the limits at which a program can perform a specific algorithm. One important `
* **Learned Dynamic Chunks (`|`):**
  ```text
  In |computer |s|cience |a|nd artificial| |intelligence, |linear| |al|gebra |is |a| computational |fr|amework |th|at all|ows |the sof|tware |to |determine |the limits |a|t |whi|ch a| program |ca|n perform |a| specific |al|gorithm. |On|e important 
  ```

#### 2. History (`6.11 bytes/chunk` — Main Transformer skipped on `83.6%` of bytes)
* **Generated Text:**
  > `During the Industrial Revolution, the invention of the steam engine and invention of the motorcycle was the first step to industrial economic growth. The industrial revolution was an era of great technological advancement and the rise of the professional structure which enabled the growth of the industrial middle class. The factors that led to th`
* **Learned Dynamic Chunks (`|`):**
  ```text
  During| |the |Industrial| |Revolution, |the invention |of |the ste|am |engine |a|nd invention |of |the motor|cycle |was |the fir|st step |to |industrial |economic |growth. |The industrial| |revolution |was |a|n |e|ra |of |great |technological| |advancement |a|nd the rise |of |the professional |structure |whi|ch enabled |the |growth |of |the industrial| |mi|ddle |cl|ass. |The factors |th|at led |to |th
  ```
  *Single chunks include whole words and multi-word phrases up to 18 bytes long:* `|the invention |`, `|industrial |economic |growth. |`, `|The industrial| |revolution |`, `|technological| |advancement |`, `|nd the rise |`, `|the professional |`, `|structure |`, `|ch enabled |`.

#### 3. Molecular Biology — Adaptive Acronym vs. Word Chunking (`4.50 bytes/chunk` — `77.8%` skipped)
* **Generated Text:**
  > `In molecular biology, DNA replication is the process by which a specific cell-to-cell conversion takes place. The complex molecules like DNA make up a large part of the cell walls of all plant living organisms.`
  > `Although there are many different forms of DNA, the process of replicating DNA has always been fascinating in its own way. In fact,`
* **Learned Dynamic Chunks (`|`):**
  ```text
  In |mo|lecular |bi|ology, |D|N|A |replication |is |the process |by |whi|ch a| specific |cell|-|to-|cell |conversion |t|akes |place. |The complex |mo|lecules |like| |D|N|A |make |up |a| large |part |of |the cell |wa|lls |of |all |plant |living |o|r|ganisms.|
  |Although |ther|e are |man|y different |for|ms |of |D|N|A, |the process |of |replicating |D|N|A |ha|s always |been |fascinating |in |i|t|s own |way. |In |fact,
  ```
  *Notice how the router allocates compute proportionally to information density:* Every time the acronym **`DNA`** appears, the router switches to **1-byte-per-character** (`|D|N|A |`), while immediately grouping surrounding words into single chunks (`|replication |`, `|the process |`, `|cell|-|to-|cell |`, `|conversion |`, `|replicating |`, `|fascinating |`). Same behavior on `(NLP)` (`|Natural| |language |processing| |(|N|L|P|) |`).

#### 4. Unseen Snake-Case Identifier (`hyper_flux_capacitor_v2`, `5.09 bytes/chunk` — `80.4%` skipped)
* **Generated Text:**
  > `When configuring the hyper_flux_capacitor_v2, engineers noticed that the hyper_flux_capacitor_v2 was the first to move around the sensor. However, since the sensor would be moved around the field, the experiment continued to move around the field and the station sensitivity would also increase. The sensor consisted of three shafts with a number of pillars inside each. The se`
* **Learned Dynamic Chunks (`|`):**
  ```text
  When |configuring |the h|yper|_|flu|x|_|capacitor|_|v|2, |engineers |noticed| |th|at the h|yper|_|flu|x|_|capacitor|_|v|2 |was |the fir|st to| |move |around |the sensor. |Ho|wever, |sin|ce |the sensor |wou|ld be |moved |around |the field, |the experiment |continued |to |move |around |the field |a|nd the station |sensitivity |wou|ld also |increase. |The sensor |consisted |of |th|ree |sh|a|fts |with |a| number |of |pi|llars |inside| |ea|ch. |The se
  ```

---

## v2 Upgrade: End-of-Chunk (`ends: true`) Causal Routing & Large Effective Batch Size (`grad_accum = 8`)

In our original 110,000-step run above (which followed the 2025 `LH-Net` formulation), you can spot the classic **1–2 byte lookahead shift** on several words (`In |computer |s|cience |a|nd`, `|fr|amework`, `|t|akes`). We investigated why this happens and how Cartesia's updated H-Net (*"From Tokens to Bytes"*, Figure 9) resolves it:

### 1. Fixing the `LH-Net` Lookahead Bug via End-of-Chunk (`ends: true`) Causal Alignment
In the 2025 `LH-Net` (`ends: false`):
- Position $t = 0$ was hardcoded to $p_0 = 1.0$, and `DeChunkLayer` indexed smoothed chunks via `cumsum(boundary_mask) - 1`.
- Because `residual_proj` was initialized to `0`, the Decoder was initially blind to byte spelling inside a chunk unless the router waited to see the **first byte of the next word** (`s` in `science`, `a` in `and`) before firing a boundary — causing the Main Transformer to be **skipped** on the highest-entropy byte of every word (the first letter after a space!).

In our updated **`hnet/dc.py`** & **`hnet/model.py`**:
1. **Unforced $t=0$ with Learned `bos_embed` ($\hat{x}_{-1}$):** Position $t=0$ computes cosine dissimilarity against a learned initial state instead of forcing $b_0 = 1$.
2. **End-of-Chunk (`ends: true`) Causal Dechunking (`init_chunk` + `cumsum(b)`):** `DeChunkLayer` prepends a learned `init_chunk` ($\tilde{z}_{-1}$) at index `0` of the EMA scan and indexes directly with `cumsum(boundary_mask)`. When chunk $k$ **finishes** at byte $t_k$ (e.g., at the trailing space `' '` of a word or end of a stem), `ChunkLayer` sends the completed word representation `h_enc[t_k]` to the Main Transformer, and `DeChunkLayer` broadcasts $\tilde{z}_k$ across positions $[t_k \dots t_{k+1}-1]$ to predict all bytes of the next chunk $[t_k+1 \dots t_{k+1}]$.
3. **Scaled-Identity Residual Initialization (`0.5 * I`):** Keeps the byte-spelling encoder pathway active from Step 1 so the router does not need to fire early inside words.
4. **Stronger Switch-Transformer Ratio Loss (`aux_loss_weight = 0.20`):** Locks empirical compression tightly without distorting relative token boundary probabilities.

---

### 2. Why Effective Batch Size (`grad_accum = 8`, `1.05 MB/step`) Matters for H-Net Routing

In Cartesia's Figure 9 interactive viewer (`3.67B` 2-stage model trained for `50,000` steps on `467.2B` bytes), each optimizer step processes **`9.34 MB/step`** (`~2.04M` subword-token equivalents per step). By contrast, a single micro-batch of `64 × 2048` on 1 GPU is only **`131 KB/step`**.

Why does effective batch size matter **even more** in H-Net than in a standard tokenized Transformer?
* **Router STE & Switch-Loss Variance:** The auxiliary compression loss $\mathcal{L}_{\text{ratio}} = \frac{N}{N-1}\big((N-1) F \cdot G + (1 - F)(1 - G)\big)$ estimates the empirical boundary fraction $F = \mathbb{E}[b_t]$ and mean probability $G = \mathbb{E}[p_t]$ over each batch. At `131 KB/step`, document-level variation (e.g., code/tables vs. prose) causes $F$ and $G$ to fluctuate from step to step (`3.7x`–`4.4x`), injecting gradient noise into the cosine router's $W_q, W_k$ projections.
* **Sweet Spot for Single-GPU Pretraining (`--grad-accum 8`, `1.05 MB/step`):**
  Accumulating over `8` micro-batches (`64 × 2048 × 8 = 1,048,576 bytes/step` $\approx$ `230k` chunks/step) cuts router gradient noise by $\sqrt{8} \approx 2.8\times$ while still giving AdamW `10,500` updates over a 10-hour (`11.0B`-byte) single-GPU run.

```bash
python train.py \
  --preset hnet_1stage_l12 \
  --dataset fineweb-edu \
  --seq-len 2048 \
  --batch-size 64 \
  --grad-accum 8 \
  --max-steps 10500 \
  --warmup-steps 500 \
  --lr 4.5e-4 \
  --log-every 5 \
  --eval-every 250 \
  --save-dir checkpoints_ga8
```

| Architecture & Effective Batch Size | Bytes to Reach `< 1.18` EMA BPB | Bytes to Beat Old `110k` Final Loss (`1.050` BPB) | Best EMA / Batch BPB | Step-to-Step Compression Stability |
| :--- | :---: | :---: | :---: | :---: |
| **Original 2025 `LH-Net` (`131 KB/step`, `grad_accum=1`)** | `5.24 GB` (Step 40,000) | `14.42 GB` (Step 110,000) | `1.0076` EMA / `0.9118` Batch | Drifts `3.50x–4.40x` |
| **New `ends: true` H-Net (`131 KB/step`, `grad_accum=1`)** | `2.76 GB` (Step 21,000) | **`6.24 GB` (Step 47,600)** | **`0.9625` EMA / `0.8405` Batch** | `3.85x–4.35x` |
| **New `ends: true` H-Net (`1.05 MB/step`, `grad_accum=8`)** | **`3.15 GB` (Step 3,000)** | **`6.80 GB` (Step 6,500)** | **`0.9919` EMA / `0.9475` (`1.05 MB` Batch)** | **Locked `4.03x–4.35x`** |

#### Full 10-Hour `grad_accum = 8` Scaling Curve (`logs/train_10h_ga8.log`, `10,500` Steps / `11.01B` Bytes)

| Step | Raw Text Processed | `1.05 MB` Batch BPB | Smoothed EMA BPB | Cross-Entropy (nats) | Empirical Compression | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.05 MB | `8.0698` | `8.0698` | `5.5935` | `2.09 B/chunk` | `9.00e-07` |
| **500** | 0.52 GB | `1.8040` | `1.8514` | `1.2504` | `4.51 B/chunk` | `4.50e-04` |
| **1,000** | 1.05 GB | `1.5804` | `1.5612` | `1.0954` | `4.38 B/chunk` | `4.47e-04` |
| **2,000** | 2.10 GB | `1.2921` | `1.2823` | `0.8956` | `4.05 B/chunk` | `4.28e-04` |
| **3,000** | 3.15 GB | `1.1763` | `1.1759` | `0.8154` | `4.14 B/chunk` | `3.91e-04` |
| **4,000** | 4.19 GB | `1.1721` | `1.1480` | `0.8124` | `3.99 B/chunk` | `3.39e-04` |
| **5,000** | 5.24 GB | `1.1306` | `1.1280` | `0.7836` | `4.14 B/chunk` | `2.76e-04` |
| **6,000** | 6.29 GB | `1.0653` | `1.0689` | `0.7384` | `4.13 B/chunk` | `2.10e-04` |
| **7,000** | 7.34 GB | `1.0157` | `1.0372` | `0.7040` | `4.28 B/chunk` | `1.47e-04` |
| **8,000** | 8.39 GB | `1.0766` | `1.0423` | `0.7462` | `4.08 B/chunk` | `9.45e-05` |
| **9,490** | 9.95 GB | **`0.9475`** *(min)* | `1.0020` | `0.6568` | `4.31 B/chunk` | `5.53e-05` |
| **10,150** | 10.64 GB | `0.9712` | **`0.9919`** *(min)* | `0.6732` | `4.29 B/chunk` | `4.63e-05` |
| **10,500** | **11.01 GB** | **`1.0605`** | **`1.0449`** | `0.7351` | **`4.04 B/chunk`** | `4.50e-05` |

---

### 3. Chunking Evolution & Final Results (`grad_accum = 8`, `Step 10,500` / `11.01B` Bytes)

#### A. Direct Comparison on Cartesia's Figure 9 Abstract (`Step 2,000` $\rightarrow$ `Step 10,500`)
Compare our **151M 1-stage** model (`grad_accum = 8`) against Cartesia's **3.67B 2-stage** model Stage 0 (`14.0B` bytes, $24\times$ more parameters):

* **Our 151M Model @ Step 2,000 (`2.10B` bytes, `4.92 B/chunk`, `52 chunks`):**
  ```text
  Maj|or |progress |on |language |models |(|L|M|s|)| |in |recent |years |has |largely |resulted |from |moving |away |from |specialized |models |designed |for |specific |ta|sks, |to| general |models |based |on |powerful |arc|hitectures |(|e.|g. |the |T|rans|former)| |that |learn |every|thing |from |raw |data.
  ```
* **Our 151M Model @ Step 6,000 (`6.29B` bytes) through Step 10,500 (`11.01B` bytes, `5.12 B/chunk`, `50 chunks`):**
  ```text
  Major |progress |on |language |models |(|L|M|s)| |in |recent |years |has |largely |resulted |from |moving |away |from |specialized |models |designed |for |specific |ta|sks, |to| general |models |based |on |powerful |arc|hitectures |(|e.|g. |the |T|rans|former)| |that |learn |every|thing |from |raw |data.
  ```
* **Cartesia 3.67B Model Stage 0 (Figure 9, Step 1,494 — `14.0B` bytes, `5.70 B/chunk`):**
  ```text
  Major |progress |on |language| |models |(LMs) |in |recent |year|s |has| |largely resulted |from| |moving |away |from| |specialized |models |designed |for| |specific |task|s, |to |general |models |based |on |power|ful| |architectures |(e.g. |the| |Transformer) |that| |learn |every|thing| |from| |raw |data.
  ```
  *Notice:*
  * By Step 6,000 (`6.29B` bytes), even the opening word `Maj|or ` merges into `Major |`, and the chunking remains **100% invariant** through `best.pt` (`Step 9,500`) and `step_10500.pt` (`Step 10,500`).
  * **32 out of 35 non-parenthetical words** in the abstract are exact 1-word chunks (`Major |progress |on |language |models |`, `|in |recent |years |has |largely |resulted |from |moving |away |from |specialized |models |designed |for |specific |`, `|to| general |models |based |on |powerful |`, `|that |learn |... |from |raw |data.`), and the remaining 3 are clean morphological/stem splits (`ta|sks, `, `arc|hitectures `, `every|thing `).

#### B. Final Generated Samples at Step 10,500 (`11.01B` Bytes, `grad_accum = 8`, `EMA BPB: 1.0449`)

1. **Computer Science & AI (`6.15 B/chunk` — Main Transformer skipped on `83.7%` of bytes):**
   * **Generated Text:**
     > `In computer science and artificial intelligence, as well as the development of creative organizations, enterprises, and the professional sector as a whole, to ensure an environment in which innovation and creativity can flourish. The reduction in the waste of energy and energy and the rem`
   * **Learned Dynamic Chunks (`|`):**
     ```text
     In |computer |science |and |arti|ficial |intelligence, |as |well |as |the |development |of |creative |organizations, |enterprises, |and |the |professional |sector |as |a |whole, |to| ensure |an |environment |in |which |inn|ovation |and |creativity |can |flo|urish.| The |reduction |in |the |waste |of |energy |and |energy |and |the |rem
     ```
     *22 consecutive single-word chunks (`|intelligence, |as |well |as |the |development |of |creative |organizations, |enterprises, |and |the |professional |sector |as |a |whole, |to| ensure |an |environment |in |which |`) — including multi-syllable words like `development `, `creative `, `organizations, `, `enterprises, `, `professional `, and `environment `.*

2. **History (`5.40 B/chunk` — Main Transformer skipped on `81.5%` of bytes):**
   * **Generated Text:**
     > `During the Industrial Revolution, the invention of the steam engine and the experimental interface for the production of computers and the distribution of business made possible the evolution of the steam engine and the steam engine to the great powers of man and the progress of the modern mind to the great`
   * **Learned Dynamic Chunks (`|`):**
     ```text
     Du|ring |the |I|ndustrial |Revolution, |the |invention |of |the |ste|am |engine |and |the |experimental |interf|ace |for |the |production |of |computers |and |the |distribution |of |business |made |possible |the |evolution |of |the |ste|am |engine |and |the |ste|am |engine |to| the |great |powers |of |man |and |the |progress |of |the |modern |mind |to| the |great
     ```
     *Two 16-word streaks of exact single-word chunks (`|for |the |production |of |computers |and |the |distribution |of |business |made |possible |the |evolution |of |the |` and `|to| the |great |powers |of |man |and |the |progress |of |the |modern |mind |to| the |great`).*

3. **Molecular Biology — Acronym (`D|N|A|`) & Rare Greek Medical Terms (`cho|n|dro|cy|tes,`) vs. Common Words (`4.19 B/chunk`):**
   * **Generated Text:**
     > `In molecular biology, DNA replication is the process by which cells germinate and expand from a single cell to form an embryo. Here we report one of the first steps in this process. The chondrocytes, a type of white blood cell that carries oxygen to the tissues, form a line of cells that contain a sin`
   * **Learned Dynamic Chunks (`|`):**
     ```text
     In |molecular |bi|ology, |D|N|A| |repl|ication |is |the |process |by |which |cells |ger|minate |and |expand |from |a |single |cell |to| form |an |emb|ryo.| Here| |we |report |one |of |the |first |ste|ps |in |this |process.| The |cho|n|dro|cy|tes, |a |type |of |whit|e |blood |cell |that |car|ries |ox|y|gen |to| the |tissues, |form |a |line |of |cells |that |contain |a |sin
     ```
     *Adaptive compute allocation in action: splits the acronym `|D|N|A| |` character-by-character, decomposes the rare Greek medical term `|cho|n|dro|cy|tes, |` and `|ox|y|gen |` into their exact Greek syllables, and immediately chunks common English words (`|to| the |tissues, |form |a |line |of |cells |that |contain |a |`) as whole words.*

4. **Unseen Snake-Case Identifier (`hyper_flux_capacitor_v2` $\rightarrow$ `v1`, `4.29 B/chunk`):**
   * **Generated Text:**
     > `When configuring the hyper_flux_capacitor_v2, engineers noticed that computer models were failing and they started to work with flat parameters. The company also implemented a flux capacitor circuit that can be used to control the flux that is delivered to the hyper_flux_capacitor v1.`
     > `The flux capacitor v2 c`
   * **Learned Dynamic Chunks (`|`):**
     ```text
     When |conf|iguring |the |hy|per|_|flu|x_|capa|citor_|v2|, |engineers |noticed |that |computer |models |were |failing |and |they |started |to| work |with |fla|t |para|meters.| The |company |also |implemented |a |flu|x |capa|citor |cir|cuit |that |can |be |used |to| control |the |flu|x |that |is |del|ivered |to| the |hy|per|_|flu|x_|capa|citor |v1|.|
     The |flu|x |capa|citor |v2| |c
     ```


