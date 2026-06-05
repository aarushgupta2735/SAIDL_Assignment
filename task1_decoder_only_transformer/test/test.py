"""
test.py
-------
Final evaluation on the held-out test set.

Run this ONLY after all training and model selection is complete.
This file never touches train or val data — it only loads a saved
checkpoint and evaluates it on wiki.test.txt.

Usage:
    python -m test.test --run_id standard_sinusoidal_T1024_L2_H4_C256

The run_id must match a folder under experiments/checkpoints/ that
contains best.pt and config.json saved during training.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
import argparse
import torch
import tiktoken
import wandb

from task1_decoder_only_transformer.src.transformer import Transformer
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.evaluate.validate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Final test set evaluation.")
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="Run ID matching a folder under experiments/checkpoints/"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best",
        choices=["best", "last"],
        help="Which checkpoint to evaluate (default: best)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")

    # --- Load config from checkpoint folder ---
    # Config is saved by logger during training so test.py never needs to
    # manually reconstruct it — avoids any mismatch with the trained model
    ckpt_dir = os.path.join("experiments", "checkpoints", args.run_id)
    config_path = os.path.join(ckpt_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            f"Check that run_id is correct and training has completed."
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    config = TransformerConfig(**config_dict)
    print(f"Loaded config for run: {args.run_id}")

    # --- Load checkpoint ---
    ckpt_path = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

    # --- Rebuild model and load weights ---
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Load test data ---
    # Test data loaded here and nowhere else in the codebase
    tokenizer = tiktoken.get_encoding("gpt2")
    data_dir  = os.path.join(os.path.dirname(__file__), "../data")

    with open(os.path.join(data_dir, "wiki.test.txt"), encoding="utf-8") as f:
        test_data = f.read()

    test_tokens = torch.tensor(
        tokenizer.encode(test_data), dtype=torch.long, device=device
    )
    print(f"Test tokens: {len(test_tokens):,}")

    # --- Evaluate ---
    print("Evaluating on test set...")
    test_loss = evaluate(model, test_tokens, config, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f"\nTest loss : {test_loss:.4f}")
    print(f"Test PPL  : {test_ppl:.2f}")

    # --- Log to wandb run ---
    # Resume the existing wandb run so test results appear alongside
    # training curves in the same run, not as a separate run
    run = wandb.init(
        project = "saidl-core-ml",
        id      = checkpoint.get("wandb_run_id", None),
        name    = args.run_id,
        resume  = "allow",
    )
    # Log as summary so it appears as a single scalar in the runs table
    wandb.run.summary["test/loss"]       = test_loss
    wandb.run.summary["test/perplexity"] = test_ppl
    wandb.finish()

    print(f"\nTest results logged to wandb run: {args.run_id}")
    print("You can now run evaluate/summarise.py to generate your report tables.")


if __name__ == "__main__":
    main()