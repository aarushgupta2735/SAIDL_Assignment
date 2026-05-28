# evaluation/run_eval.py
# run this after training to evaluate all saved checkpoints

import os
import json
import torch
from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from evaluation.evaluate import evaluate

CHECKPOINTS_DIR = "experiments/checkpoints"
RESULTS_DIR     = "experiments/results"

def run_all(val_tokens, device):
    for experiment in os.listdir(CHECKPOINTS_DIR):
        ckpt_dir   = os.path.join(CHECKPOINTS_DIR, experiment)
        result_dir = os.path.join(RESULTS_DIR, experiment)
        os.makedirs(result_dir, exist_ok=True)

        # load config that was saved alongside the checkpoint
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg_dict = json.load(f)

        config = TransformerConfig(**cfg_dict)
        model  = Transformer(config).to(device) 
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "model.pt"))
        )

        metrics = evaluate(model, val_tokens, config, device)
        metrics["experiment"] = experiment

        # save summary
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"{experiment:30s} | "
              f"ppl {metrics['perplexity']:>8.2f} | "
              f"gpu {metrics['peak_gpu_mb']:>7.1f}mb | "
              f"tok/s {metrics['tokens_per_sec']:>8.0f}")