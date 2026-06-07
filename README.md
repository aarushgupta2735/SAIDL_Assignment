# SAiDL Summer Induction Assignment 2026

**Author:** Aarush Gupta — 2023B3A70839G  
**Tasks completed:** Core ML (Task 1) · Reinforcement Learning (Task 2)

---

## Repository Structure

```
SAiDL-Summer-Assignment2026/
├── run.py                              # Single entry point for all runs
├── environment.yml                     # Conda environment
├── Final Report Merged.pdf             # Combined report (both tasks)
├── Task 1 Report.pdf                   # Core ML report
├── Task 2 Report.pdf                   # RL report
│
├── task1_decoder_only_transformer/     # Core ML — decoder-only transformer
│   ├── config/
│   │   ├── transformer_config.py       # TransformerConfig dataclass
│   │   └── train_config.py             # TrainConfig dataclass
│   ├── src/
│   │   ├── transformer.py              # Top-level Transformer model
│   │   ├── decoder_layer/
│   │   │   ├── decoder.py              # Stacks N DecoderBlocks
│   │   │   ├── decoder_block.py        # One block: attention + FFN + residuals
│   │   │   ├── multi_self_decoder.py   # Registry-based multi-head attention
│   │   │   ├── mqa_multi_self_decoder.py
│   │   │   └── single_decoder/
│   │   │       ├── standard_single_self_decoder.py
│   │   │       ├── local_single_self_decoder.py
│   │   │       ├── block_sparse_single_self_decoder_block.py
│   │   │       └── mqa_single_self_decoder.py
│   │   ├── data_prep/
│   │   │   ├── data_prep.py            # Embedding + PE registry
│   │   │   ├── vector_embedding.py
│   │   │   └── positional_embedding/
│   │   │       ├── standard_positional_embedding.py   # Sinusoidal
│   │   │       ├── rotatory_positional_embedding.py   # RoPE
│   │   │       ├── attention_positional_embedding.py  # ALiBi
│   │   │       └── relative_positional_embedding.py   # Shaw et al.
│   │   ├── feedforward.py
│   │   └── layerNorm.py
│   ├── evaluate/
│   │   ├── logger.py                   # WandB logger
│   │   ├── validate.py                 # Validation loop
│   │   └── summarise.py
│   ├── train/
│   │   ├── train.py
│   │   └── loop.py
│   ├── data/
│   │   ├── wiki.train.txt
│   │   ├── wiki.valid.txt
│   │   └── wiki.test.txt
│   ├── experiments/checkpoints/        # Saved model checkpoints
│   └── generate.py                     # Text generation script
│
└── task2_TD3/                          # RL — TD3 + Transformer actor
    ├── config/
    │   └── config.py                   # TD3config + TransformerConfig
    ├── src/
    │   ├── Actor.py                    # MLP or Transformer actor
    │   ├── Critic.py                   # Twin MLP critics
    │   ├── ReplayBuffer.py             # Circular replay buffer with history
    │   ├── TD3.py                      # TD3 algorithm
    │   └── transformer.py              # Causal transformer actor
    ├── evaluate/
    │   └── rl_logger.py                # WandB logger for RL
    └── train/
        └── train.py                    # Training loop
```

---

## Task 1 — Core ML: Long-Context Sequence Modelling

### Overview

A modular decoder-only Transformer trained on WikiText-2 for language modelling. Attention mechanisms, positional encodings, and convolutional components can be swapped independently via config — no model code changes required.

### What was implemented

**Attention variants:**
- Standard scaled dot-product attention with causal mask
- Sliding window (local) attention — each chunk attends to itself and the previous chunk
- Sparse block attention — within-block attention only, no cross-block
- Multi-Query Attention (MQA) — shared K/V across all heads

**Positional encodings:**
- Sinusoidal (absolute) — added to embeddings in DataPrep
- RoPE — applied to Q and K inside each attention head
- ALiBi — linear bias added to attention scores inside each head
- Relative PE (Shaw et al.) — learned embedding table indexed by relative distance

**Conv–Attention hybrids:**
- Pre-attention Conv1D — 1D convolution before each attention block
- Interleaved Conv1D — alternating conv and attention layers

**PE placement rule:**
- Sinusoidal → `DataPrep` (added to token embeddings)
- RoPE / ALiBi / Relative → inside attention head (modifies Q, K, or scores)
- `DataPrep` returns `nn.Identity` for the latter three

### Key design decisions

- Weight tying between input embedding and output linear projection (`self.lin.weight = self.data_prep.ve.embed.weight`)
- Registry pattern for attention and PE — config fields `attention` and `positional_encoding` drive instantiation
- `TransformerConfig.run_id()` auto-generates run names like `standard_sinusoidal_T1024_L2_H4_C64`
- `bfloat16` autocast + `torch.compile` on Linux for throughput
- Cosine LR schedule with warmup

### Results summary

| Attention | Context | Val PPL | Peak GPU (MB) | Throughput (tok/s) | Latency (ms) |
|-----------|---------|---------|--------------|-------------------|--------------|
| Standard  | 512     | 106.24  | 10,503       | 192.61            | 664.54       |
| Local     | 512     | **100.35**  | 10,341   | 93.18             | 1,373.62     |
| MQA       | 512     | 108.30  | 10,472       | 217.45            | 588.64       |
| Standard  | 1024    | 106.06  | 21,726       | 179.24            | 714.13       |
| Local     | 1024    | **97.94**   | 20,622   | 67.74             | 1,889.54     |
| MQA       | 1024    | 106.78  | 21,663       | 209.55            | 610.82       |
| Sparse    | 1024    | 107.83  | 20,241       | 186.00            | 688.16       |

| PE        | Val PPL | Peak GPU (MB) | Throughput (tok/s) | Latency (ms) |
|-----------|---------|--------------|-------------------|--------------|
| Sinusoidal| 107.20  | 10,503       | 264.65            | 483.65       |
| ALiBi     | 105.26  | 10,510       | 230.96            | 554.20       |
| Relative  | 82.88   | 10,587       | 208.48            | 613.96       |
| **RoPE**  | **81.89** | 10,502     | 172.51            | 741.98       |

### Running Task 1

Edit `run.py` to call `main_decoder_only_transformer()`, then:

```bash
python run.py
```

To change experiment config, edit the `TransformerConfig` and `TrainConfig` instantiation in `task1_decoder_only_transformer/train/train.py`:

```python
config = TransformerConfig(
    attention="standard",           # standard | local | sparse | mqa
    positional_encoding="rope",     # sinusoidal | rotatory | relative | attention
    use_conv=False,
    conv_type="none",               # none | pre_attn | interleaved
    context_window=1024,
    n_decoder_layers=2,
    n_heads=4,
    embedding_size=64,
    dropout=0.3,
)
```

---

## Task 2 — Reinforcement Learning: Transformer Actors in TD3

### Overview

TD3 with MLP and Transformer actors trained on Hopper-v5 (gymnasium). The Transformer actor takes a sliding window of the last `L` observation-action pairs as input and outputs an action from the final position. Four POMDP settings are tested: hidden velocities, observation noise, delayed rewards, and a combined setting.

### What was implemented

**TD3 baseline (MLP):**
- Actor: `Linear(11,256) → ReLU → Linear(256,256) → ReLU → Linear(256,3) → Tanh`
- Critics: twin `Linear(14,256) → ReLU → Linear(256,256) → ReLU → Linear(256,1)`
- Clipped double-Q, delayed policy update, target policy smoothing, Polyak averaging

**Transformer actor:**
- Input projection: `Linear(obs + act, embed_dim)`
- `N` causal `DecoderBlock` layers (imported from Task 1)
- Output: action from last sequence position only — `x[:, -1, :]`
- Pre-LN for training stability

**ReplayBuffer:**
- Pre-allocated circular buffer on GPU
- `get_history(idx, L)` walks backwards collecting `(obs, action)` pairs, stops at episode boundary or env-id change, pads with zeros

**POMDP settings** (each independently configurable via `TD3config` flags):
- `is_velocity_hidden` — removes velocity dimensions from observations
- `add_observation_noise` — Gaussian noise `N(0, σ²I)` added to every observation
- `delay_rewards` — accumulates rewards for `K` steps, zeros otherwise
- Combined POMDP — all three simultaneously (MLP only)

### Results summary

| Condition | Agent | Best Return | Final Return |
|-----------|-------|------------|--------------|
| Full Obs | MLP-TD3 | 1121.6 | 1313.8 |
| Full Obs (L=4) | TF-TD3 | 23.4 | 14.3 |
| Hidden Vel | MLP-TD3 | 38.0 | 38.3 |
| Hidden Vel | TF-TD3 (L=8) | 25.8 | 21.2 |
| Obs Noise σ=0.3 | MLP-TD3 | 711.9 | 465.4 |
| Obs Noise σ=0.3 | TF-TD3 (L=8) | 22.5 | 23.1 |
| Delayed K=10 | MLP-TD3 | 3002.6 | 1640.8 |
| Delayed K=10 | TF-TD3 (L=8) | 25.2 | 9.5 |
| Combined POMDP | MLP-TD3 | 316.1 | 315.3 |

### Running Task 2

Edit `run.py` to call `main_td3()`, then:

```bash
python run.py
```

Key config flags in `task2_TD3/config/config.py`:

```python
config = TD3config(
    use_transformer=False,          # True for Transformer actor
    is_velocity_hidden=False,       # POMDP: hide velocity dims
    add_observation_noise=False,    # POMDP: Gaussian obs noise
    observation_noise_std=0.3,
    delay_rewards=False,            # POMDP: accumulate rewards
    K_delayed_rewards=10,
    training_iterations=1_000_000,
    n_envs=3,
)
```

---

## Setup

### Conda environment

```bash
conda env create -f environment.yml
conda activate saidl2026
```

### Manual install (pip)

```bash
pip install torch torchvision torchaudio
pip install "gymnasium[mujoco]"
pip install wandb tiktoken numpy
```

### WandB

```bash
wandb login
```

Experiments are tracked under projects `saidl-core-ml` (Task 1) and `saidl-rl` (Task 2).

### Platform notes

| Platform | torch.compile | Recommended for |
|----------|--------------|-----------------|
| Windows  | ❌ (no Triton) | Development only |
| Linux / WSL | ✅ | Full training runs |
| Lightning AI / Colab | ✅ | Cloud training |

On Windows, `torch.compile` is automatically disabled via platform detection in `TD3.__init__`. TF32 flags are always enabled where supported:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## Hardware used

| Task | Hardware | Notes |
|------|----------|-------|
| Task 1 (T=512) | NVIDIA A100 | Lightning AI |
| Task 1 (T=1024) | NVIDIA L40S | Lightning AI |
| Task 2 MLP runs | RTX 5090 + i9 14th gen | Local Windows |
| Task 2 Transformer runs | NVIDIA A100 | Lightning AI, Linux |

---

## Experiment tracking

All runs logged to Weights & Biases. Metrics tracked:

**Task 1:** train/val loss, val perplexity, peak GPU memory (MB), throughput (tokens/sec), inference latency (ms/seq), gradient norm

**Task 2:** episode return (train + eval), critic loss (phi1, phi2), actor loss, buffer size, steps/sec

Checkpoints saved to `task1_decoder_only_transformer/experiments/checkpoints/{run_id}/`.

---

## Known limitations

- Extrapolation test (Task 1) incomplete — index errors at T>512 for RoPE and Relative PE due to buffer size; ALiBi and Sinusoidal failures were not isolated
- Conv hybrid validation perplexity reported as ~1.0 (bug in validation loop for conv variants — throughput numbers are valid, PPL is not)
- Transformer actor (Task 2) did not learn across any condition — likely due to fragmented replay buffer history and online RL training instability
- All Task 2 results are single-seed; 3 seeds required per assignment spec but not achievable within compute budget
- RLHF not implemented