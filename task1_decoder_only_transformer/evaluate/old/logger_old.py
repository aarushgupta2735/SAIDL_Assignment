"""
Central experiment logger.
Wraps wandb and handles local checkpoint organisation.
All training and evaluation code imports from here — never import wandb directly.
"""

import os
import time
import math
import json
import torch
import wandb
from typing import Optional
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class ExperimentLogger:
    def __init__(self, config: TransformerConfig, train_config: TrainConfig):
        self.config = config
        self.train_config = train_config
        self.run_id = config.run_id()

        # --- Local checkpoint dir ---
        self.checkpoint_dir = os.path.join("experiments", "checkpoints", self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Save config locally so every run is reproducible ---
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # --- wandb init ---
        self.run = wandb.init(
            project="saidl-core-ml",
            name=self.run_id,
            group=config.experiment_name,   # groups related runs (e.g. "attention_variants")
            config={**config.to_dict(), "iterations": train_config.iterations,
                    "warmup_steps": train_config.warmup_steps},
            resume="allow",
        )

        # Timing state
        self._iter_start: Optional[float] = None
        self._tokens_per_iter = config.batch_size * config.context_window

    # ------------------------------------------------------------------
    # Training loop hooks
    # ------------------------------------------------------------------

    def on_iter_start(self):
        """Call at the top of each training iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._iter_start = time.perf_counter()

    def on_iter_end(
        self,
        i: int,
        train_loss: float,
        lr: float,
        grad_norm: float,
    ):
        """Call at the end of each training iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._iter_start
        throughput = self._tokens_per_iter / elapsed  # tokens/sec

        metrics = {
            "train/loss":       train_loss,
            "train/perplexity": math.exp(min(train_loss, 20)),  # cap to avoid overflow
            "train/lr":         lr,
            "train/grad_norm":  grad_norm,
            "perf/throughput_tokens_per_sec": throughput,
            "perf/iter_time_ms": elapsed * 1000,
        }
        wandb.log(metrics, step=i)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def log_validation(self, i: int, val_loss: float):
        """Call after running validation."""
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

        metrics = {
            "val/loss":           val_loss,
            "val/perplexity":     math.exp(min(val_loss, 20)),
            "perf/peak_gpu_mb":   peak_mem_mb,
        }
        wandb.log(metrics, step=i)
        print(
            f"[{i:>6}] val_loss={val_loss:.4f}  "
            f"val_ppl={math.exp(min(val_loss,20)):.2f}  "
            f"peak_gpu={peak_mem_mb:.0f}MB"
        )

    # ------------------------------------------------------------------
    # Inference latency (run once after training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_inference_latency(self, model: torch.nn.Module, device: torch.device):
        """
        Measures single-sample autoregressive inference latency.
        Generates 128 tokens from a prompt of length 64, repeated 20 times.
        """
        model.eval()
        prompt_len, gen_len, repeats = 64, 128, 20

        dummy_input = torch.randint(0, self.config.vocab_size, (1, prompt_len), device=device)
        generated = dummy_input.clone()

        # Warmup
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate_next_token(generated)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            seq = dummy_input.clone()
            for _ in range(gen_len):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    next_tok = model.generate_next_token(seq)
                seq = torch.cat([seq, next_tok], dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_tokens = gen_len * repeats
        elapsed_tokens = (time.perf_counter() - start)
        latency_ms =  elapsed_tokens* 1000 / repeats
        tokens_per_sec = total_tokens / elapsed_tokens  

        wandb.log({
            "inference/latency_ms_per_sequence": latency_ms,
            "inference/tokens_per_sec": tokens_per_sec,
        })
        print(f"Inference latency: {latency_ms:.1f}ms/seq  |  {tokens_per_sec:.0f} tok/s")
        model.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, model: torch.nn.Module, i: int, val_loss: float):
        path = os.path.join(self.checkpoint_dir, f"ckpt_step{i}.pt")
        torch.save({
            "step":       i,
            "val_loss":   val_loss,
            "model":      model.state_dict(),
            "run_id":     self.run_id,
        }, path)

    def save_best(self, model: torch.nn.Module, val_loss: float):
        """Overwrites best.pt whenever validation loss improves."""
        path = os.path.join(self.checkpoint_dir, "best.pt")
        torch.save({
            "val_loss": val_loss,
            "model":    model.state_dict(),
            "run_id":   self.run_id,
        }, path)

    def finish(self):
        wandb.finish()