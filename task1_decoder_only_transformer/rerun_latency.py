"""
rerun_latency.py
----------------
Re-runs inference latency measurement for finished runs and logs
results back to their existing wandb runs.

Use this when:
- prompt_len was wrong during original training (e.g. 64 instead of context_window)
- You want to rerun latency on a different machine

Usage:
    python -m evaluate.rerun_latency

Steps:
1. Go to wandb -> your project -> filter group "attenti"
2. Click each run -> copy the run ID from the URL
   e.g. https://wandb.ai/<entity>/<project>/runs/<RUN_ID>
3. Fill in the runs list below with run_id and wandb_run_id
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import math
import torch
import wandb

from src.transformer import Transformer
from config.transformer_config import TransformerConfig


# ---------------------------------------------------------------
# FILL THESE IN — one entry per run you want to rerun latency for
# run_id      : folder name under experiments/checkpoints/
# wandb_run_id: the ID from the wandb run URL
# ---------------------------------------------------------------
RUNS = [
    {"run_id": "standard_attention_T1024_L2_H4_C64",  "wandb_run_id": "<paste_from_wandb_url>"},
    {"run_id": "local_attention_T1024_L2_H4_C64",     "wandb_run_id": "<paste_from_wandb_url>"},
    {"run_id": "sparse_attention_T1024_L2_H4_C64",    "wandb_run_id": "<paste_from_wandb_url>"},
    {"run_id": "mqa_attention_T1024_L2_H4_C64",       "wandb_run_id": "<paste_from_wandb_url>"},
    # add more as needed
]

WANDB_PROJECT = "saidl-core-ml"
WANDB_ENTITY  = None  # set to your wandb username if needed
GEN_LEN       = 128
REPEATS       = 20
# ---------------------------------------------------------------


@torch.no_grad()
def run_latency(model, config, device):
    model.eval()
    device_type = device.type
    prompt_len  = config.context_window  # CHANGE: use full context_window, safe for all attention types

    dummy_input = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    # Warmup
    for _ in range(3):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _ = model.generate_next_token(dummy_input)

    if device_type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(REPEATS):
        seq = dummy_input.clone()
        for _ in range(GEN_LEN):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                next_tok = model.generate_next_token(seq)
            seq = torch.cat([seq, next_tok], dim=1)

    if device_type == "cuda":
        torch.cuda.synchronize()

    elapsed       = time.perf_counter() - start
    latency_ms    = elapsed * 1000 / REPEATS
    tokens_per_sec = (GEN_LEN * REPEATS) / elapsed

    model.train()
    return latency_ms, tokens_per_sec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")

    for entry in RUNS:
        run_id      = entry["run_id"]
        wandb_run_id = entry["wandb_run_id"]

        if "<paste" in wandb_run_id:
            print(f"Skipping {run_id} — wandb_run_id not filled in")
            continue

        print(f"\nProcessing: {run_id}")

        # --- Load config ---
        ckpt_dir    = os.path.join("experiments", "checkpoints", run_id)
        config_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"  config.json not found at {config_path}, skipping")
            continue

        with open(config_path) as f:
            config = TransformerConfig(**json.load(f))

        # --- Load model ---
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  best.pt not found at {ckpt_path}, skipping")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = Transformer(config).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint (val_loss={ckpt['val_loss']:.4f})")

        # --- Run latency ---
        print(f"  Running latency (prompt_len={config.context_window}, gen_len={GEN_LEN}, repeats={REPEATS})...")
        latency_ms, tokens_per_sec = run_latency(model, config, device)
        print(f"  Latency: {latency_ms:.1f}ms/seq  |  {tokens_per_sec:.0f} tok/s")

        # --- Log to existing wandb run ---
        project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
        run = wandb.init(
            project = WANDB_PROJECT,
            entity  = WANDB_ENTITY,
            id      = wandb_run_id,
            resume  = "must",  # fail loudly if run_id is wrong
        )
        # overwrite previous latency metrics with corrected values
        wandb.log({
            "inference/latency_ms_per_sequence": latency_ms,
            "inference/tokens_per_sec":          tokens_per_sec,
        })
        wandb.finish()
        print(f"  Logged to wandb run: {wandb_run_id}")

    print("\nAll done. Run evaluate/summarise.py to regenerate your report tables.")


if __name__ == "__main__":
    main()