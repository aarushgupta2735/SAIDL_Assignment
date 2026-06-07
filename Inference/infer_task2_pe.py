"""
infer_task2_pe.py
-----------------
Runs inference on all Task2 positional encoding models
and logs results to wandb under group "task2_positional_encodings".

Directory structure expected:
    all_models/
        Task2_PE/
            standard_attention_T512_L2_H4_C64_run-[timestamp]-[id]/
                config.json, best.pt
            standard_relative_T512_L2_H4_C64_run-[timestamp]-[id]/
            standard_rotatory_T512_L2_H4_C64_run-[timestamp]-[id]/
            standard_sinusoidal_T512_L2_H4_C64_run-[timestamp]-[id]/

Usage:
    python -m evaluate.infer_task2_pe
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
WANDB_ENTITY   = None       # set to your wandb username if needed
WANDB_GROUP    = "task2_positional_encodings"
GEN_LEN        = 128
REPEATS        = 20

# Root directory containing Task2_PE/
ALL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "all_models")
TASK2_DIR      = os.path.join(ALL_MODELS_DIR, "Task2_PE")
# ---------------------------------------------------------------


@torch.no_grad()
def run_latency(model, config, device):
    model.eval()
    device_type = device.type
    prompt_len  = config.context_window  # safe for all PE types

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


@torch.no_grad()
def run_extrapolation(model, config, device, val_tokens):
    """
    Extrapolation test — train on T=512, evaluate perplexity on
    T=512, T=1024, T=2048 as required by the assignment.
    Returns dict of {context_len: perplexity}
    """
    from evaluate.validate import evaluate
    results = {}
    for test_len in [512, 1024, 2048]:
        # Temporarily override context_window for evaluation
        original_T = config.context_window
        config.context_window = test_len
        try:
            loss = evaluate(model, val_tokens, config, device)
            results[test_len] = math.exp(min(loss, 20))
        except Exception as e:
            print(f"    Extrapolation T={test_len} failed: {e}")
            results[test_len] = None
        config.context_window = original_T
    return results


def collect_runs():
    """
    Walk Task2_PE/ and collect all subdirectories
    that contain both config.json and best.pt.
    Handles duplicate folder names by keeping all of them.
    """
    runs = []
    if not os.path.exists(TASK2_DIR):
        print(f"Directory not found: {TASK2_DIR}")
        return runs

    for folder_name in sorted(os.listdir(TASK2_DIR)):
        folder_path = os.path.join(TASK2_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        config_path = os.path.join(folder_path, "config.json")
        ckpt_path   = os.path.join(folder_path, "best.pt")
        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            print(f"  Skipping {folder_name} — missing config.json or best.pt")
            continue
        runs.append({
            "folder_path": folder_path,
            "folder_name": folder_name,
        })
    return runs


def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({gpu_name})")

    # Load val tokens for extrapolation test
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    data_dir  = os.path.join(os.path.dirname(__file__), "..", "data")
    val_tokens = None
    val_path   = os.path.join(data_dir, "wiki.valid.txt")
    if os.path.exists(val_path):
        with open(val_path, encoding="utf-8") as f:
            val_data = f.read()
        val_tokens = torch.tensor(
            tokenizer.encode(val_data), dtype=torch.long, device=device
        )
        print(f"Val tokens loaded: {len(val_tokens):,}")
    else:
        print("WARNING: wiki.valid.txt not found — extrapolation test will be skipped")

    runs = collect_runs()
    print(f"\nFound {len(runs)} model(s) in Task2_PE/\n")

    for entry in runs:
        folder_path = entry["folder_path"]
        folder_name = entry["folder_name"]

        print(f"Processing: {folder_name}")

        # --- Load config ---
        with open(os.path.join(folder_path, "config.json")) as f:
            config_dict = json.load(f)
        config = TransformerConfig(**config_dict)

        # --- Load model ---
        ckpt     = torch.load(os.path.join(folder_path, "best.pt"), map_location=device)
        model    = Transformer(config).to(device)
        model.load_state_dict(ckpt["model"])
        val_loss = ckpt.get("val_loss", None)
        val_ppl  = math.exp(min(val_loss, 20)) if val_loss is not None else None
        print(f"  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}" if val_loss else "  val_loss=unknown")

        # --- Run latency ---
        print(f"  Running latency (prompt_len={config.context_window}, gen_len={GEN_LEN}, repeats={REPEATS})...")
        latency_ms, tokens_per_sec = run_latency(model, config, device)
        print(f"  Latency: {latency_ms:.1f}ms/seq  |  {tokens_per_sec:.0f} tok/s")

        # --- Peak memory ---
        peak_mb = run_peak_memory(model, config, device)
        if peak_mb:
            print(f"  Peak GPU memory: {peak_mb:.0f} MB")

        # --- Extrapolation test ---
        # Assignment requires: train on T=512, test on T=512, 1024, 2048
        extrap_results = {}
        if val_tokens is not None and config.context_window == 512:
            print("  Running extrapolation test (T=512/1024/2048)...")
            extrap_results = run_extrapolation(model, config, device, val_tokens)
            for t, ppl in extrap_results.items():
                print(f"    T={t}: PPL={ppl:.2f}" if ppl else f"    T={t}: failed")

        # --- Init wandb ---
        wandb_run_id = ckpt.get("wandb_run_id", None)
        run_name     = folder_name

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
            run = wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                name    = run_name,
                group   = WANDB_GROUP,
                config  = {
                    **config_dict,
                    "subtask": "pe_comparison",
                    "gpu":     gpu_name,
                },
                resume  = "allow",
            )

        # --- Log all metrics as summary scalars ---
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

        # Extrapolation results logged as separate metrics
        for t, ppl in extrap_results.items():
            if ppl is not None:
                metrics[f"extrapolation/ppl_T{t}"] = ppl

        for k, v in metrics.items():
            wandb.run.summary[k] = v

        wandb.finish()
        print(f"  Logged to wandb group: {WANDB_GROUP}\n")

    print("Task2 PE inference complete.")
    print("Run evaluate/summarise.py to generate your report tables.")


if __name__ == "__main__":
    main()