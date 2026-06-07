"""
infer_task1_attention.py
------------------------
Runs inference on all Task1 attention mechanism models (context windows 512 and 1024)
and logs results to wandb under group "task1_attention_mechanisms".

Directory structure expected:
    all_models/
        Task1_Attention/
            1024/
                local_sinusoidal_T1024_L2_H4_C64_run-[timestamp]-[id]/
                    config.json, best.pt
                mqa_sinusoidal_T1024_L2_H4_C64_run-[timestamp]-[id]/
                sparse_sinusoidal_T1024_L2_H4_C64_run-[timestamp]-[id]/
                standard_sinusoidal_T1024_L2_H4_C64_run-[timestamp]-[id]/
            512/
                (same structure)

Usage:
    python -m evaluate.infer_task1_attention
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
# CONFIG
# ---------------------------------------------------------------
WANDB_PROJECT  = "saidl-core-ml"
WANDB_ENTITY   = None        # set to your wandb username if needed
WANDB_GROUP    = "task1_attention_mechanisms"
GEN_LEN        = 128
REPEATS        = 20

# Root directory containing Task1_Attention/
ALL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "all_models")
TASK1_DIR      = os.path.join(ALL_MODELS_DIR, "Task1_Attention")
# ---------------------------------------------------------------


@torch.no_grad()
def run_latency(model, config, device):
    model.eval()
    device_type = device.type
    prompt_len  = config.context_window  # safe for all attention types

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

    elapsed        = time.perf_counter() - start
    latency_ms     = elapsed * 1000 / REPEATS
    tokens_per_sec = (GEN_LEN * REPEATS) / elapsed

    model.train()
    return latency_ms, tokens_per_sec


@torch.no_grad()
def run_peak_memory(model, config, device):
    """Measure peak GPU memory during a forward pass."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    dummy_x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_window), device=device)
    dummy_y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_window), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _ = model(dummy_x, dummy_y)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    return peak_mb


def collect_runs():
    """
    Walk Task1_Attention/1024/ and Task1_Attention/512/ and collect
    all subdirectories that contain both config.json and best.pt.
    Returns list of dicts with keys: folder_path, run_name, context_window
    """
    runs = []
    for context_len in ["1024", "512"]:
        context_dir = os.path.join(TASK1_DIR, context_len)
        if not os.path.exists(context_dir):
            print(f"Directory not found, skipping: {context_dir}")
            continue
        for folder_name in sorted(os.listdir(context_dir)):
            folder_path = os.path.join(context_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            config_path = os.path.join(folder_path, "config.json")
            ckpt_path   = os.path.join(folder_path, "best.pt")
            if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
                print(f"  Skipping {folder_name} — missing config.json or best.pt")
                continue
            runs.append({
                "folder_path":   folder_path,
                "folder_name":   folder_name,
                "context_window": int(context_len),
            })
    return runs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({gpu_name})")

    runs = collect_runs()
    print(f"\nFound {len(runs)} model(s) in Task1_Attention/\n")

    for entry in runs:
        folder_path    = entry["folder_path"]
        folder_name    = entry["folder_name"]
        context_window = entry["context_window"]

        print(f"Processing: {folder_name}")

        # --- Load config ---
        with open(os.path.join(folder_path, "config.json")) as f:
            config_dict = json.load(f)
        config = TransformerConfig(**config_dict)

        # --- Load model ---
        ckpt      = torch.load(os.path.join(folder_path, "best.pt"), map_location=device)
        model     = Transformer(config).to(device)
        model.load_state_dict(ckpt["model"])
        val_loss  = ckpt.get("val_loss", None)
        val_ppl   = math.exp(min(val_loss, 20)) if val_loss is not None else None
        print(f"  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}" if val_loss else "  val_loss=unknown")

        # --- Run latency ---
        print(f"  Running latency (prompt_len={config.context_window}, gen_len={GEN_LEN}, repeats={REPEATS})...")
        latency_ms, tokens_per_sec = run_latency(model, config, device)
        print(f"  Latency: {latency_ms:.1f}ms/seq  |  {tokens_per_sec:.0f} tok/s")

        # --- Peak memory ---
        peak_mb = run_peak_memory(model, config, device)
        if peak_mb:
            print(f"  Peak GPU memory: {peak_mb:.0f} MB")

        # --- Build wandb run name ---
        # Use folder_name as run name so it's unique and traceable
        run_name = folder_name

        # --- Check if this run already exists in wandb by name ---
        wandb_run_id = ckpt.get("wandb_run_id", None)

        # Init wandb — resume if we have the original run_id, else create new
        if wandb_run_id:
            run = wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                id      = wandb_run_id,
                resume  = "allow",
                group   = WANDB_GROUP,
                name    = run_name,
            )
        else:
            # No original run_id saved — create a fresh run for this model
            run = wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                name    = run_name,
                group   = WANDB_GROUP,
                config  = {
                    **config_dict,
                    "context_window": context_window,
                    "subtask":        "attention_comparison",
                    "gpu":            gpu_name,
                },
                resume  = "allow",
            )

        # --- Log all metrics ---
        metrics = {
            "inference/latency_ms_per_sequence": latency_ms,
            "inference/tokens_per_sec":          tokens_per_sec,
            "gpu/name":                          gpu_name,
        }
        if val_loss is not None:
            metrics["val/loss"]       = val_loss
            metrics["val/perplexity"] = val_ppl
        if peak_mb is not None:
            metrics["perf/peak_gpu_mb"] = peak_mb

        # Log as summary so they appear as scalar cells in runs table
        for k, v in metrics.items():
            wandb.run.summary[k] = v

        wandb.finish()
        print(f"  Logged to wandb group: {WANDB_GROUP}\n")

    print("Task1 attention inference complete.")
    print("Run evaluate/summarise.py to generate your report tables.")


if __name__ == "__main__":
    main()