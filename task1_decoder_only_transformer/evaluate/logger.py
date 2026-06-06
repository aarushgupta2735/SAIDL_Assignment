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
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig


class ExperimentLogger:
    def __init__(self, config: TransformerConfig, train_config: TrainConfig, device: torch.device):
        self.config = config
        self.train_config = train_config
        self.run_id = config.run_id()
        self.device = device
        self.device_type = device.type

        # --- Local checkpoint dir ---
        self.checkpoint_dir = os.path.join("experiments", "checkpoints", self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Save config locally so test.py can reconstruct the model ---
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # --- wandb init ---
        self.run = wandb.init(
            project="saidl-core-ml",
            name=self.run_id,
            group=f"{config.attention}_{config.positional_encoding}",
            config={**config.to_dict(),
                    "iterations":   train_config.iterations,
                    "warmup_steps": train_config.warmup_steps},
            resume="allow",
        )

        self._iter_start: Optional[float] = None
        self._tokens_per_iter = config.batch_size * config.context_window
        self._train_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Training loop hooks
    # ------------------------------------------------------------------

    def on_iter_start(self):
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self._iter_start = time.perf_counter()

    def on_iter_end(self, i: int, train_loss: float, lr: float, grad_norm: float):
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._iter_start
        throughput = self._tokens_per_iter / elapsed

        wandb.log({
            "train/loss":                     train_loss,
            "train/perplexity":               math.exp(min(train_loss, 20)),
            "train/lr":                       lr,
            "train/grad_norm":                grad_norm,
            "perf/throughput_tokens_per_sec": throughput,
            "perf/iter_time_ms":              elapsed * 1000,
        }, step=i)

    # ------------------------------------------------------------------
    # Epoch time (every 1000 steps)
    # ------------------------------------------------------------------

    def log_epoch_time(self, i: int, elapsed_sec: float):
        wandb.log({"perf/time_per_1000_steps_sec": elapsed_sec}, step=i)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def log_validation(self, i: int, val_loss: float):
        metrics = {
            "val/loss":       val_loss,
            "val/perplexity": math.exp(min(val_loss, 20)),
        }
        if self.device_type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
            metrics["perf/peak_gpu_mb"] = peak_mem_mb
            print(
                f"[{i:>6}] val_loss={val_loss:.4f}  "
                f"val_ppl={math.exp(min(val_loss,20)):.2f}  "
                f"peak_gpu={peak_mem_mb:.0f}MB"
            )
        else:
            print(
                f"[{i:>6}] val_loss={val_loss:.4f}  "
                f"val_ppl={math.exp(min(val_loss,20)):.2f}"
            )
        wandb.log(metrics, step=i)

    # ------------------------------------------------------------------
    # Inference latency
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_inference_latency(self, model: torch.nn.Module,config):
        model.eval()
        gen_len, repeats = 128, 20
        prompt_len = config.contex
        dummy_input = torch.randint(
            0, self.config.vocab_size, (1, prompt_len), device=self.device
        )

        # Warmup
        for _ in range(3):
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                _ = model.generate_next_token(dummy_input)

        if self.device_type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(repeats):
            seq = dummy_input.clone()
            for _ in range(gen_len):
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    next_tok = model.generate_next_token(seq)
                seq = torch.cat([seq, next_tok], dim=1)

        if self.device_type == "cuda":
            torch.cuda.synchronize()

        total_tokens = gen_len * repeats
        elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000 / repeats
        tokens_per_sec = total_tokens / elapsed
        total_train_time_min = (time.perf_counter() - self._train_start) / 60

        wandb.log({
            "inference/latency_ms_per_sequence": latency_ms,
            "inference/tokens_per_sec":          tokens_per_sec,
            "perf/total_train_time_min":         total_train_time_min,
        })
        print(f"Inference latency: {latency_ms:.1f}ms/seq  |  {tokens_per_sec:.0f} tok/s")
        print(f"Total training time: {total_train_time_min:.1f} min")
        model.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # CHANGE: wandb_run_id now saved in every checkpoint so test.py can
    # resume the correct wandb run and log test metrics to it
    # ------------------------------------------------------------------

    def save_last(self, model: torch.nn.Module, i: int, val_loss: float):
        path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save({
            "step":         i,
            "val_loss":     val_loss,
            "model":        model.state_dict(),
            "run_id":       self.run_id,
            "wandb_run_id": self.run.id,      # CHANGE
        }, path)

    def save_best(self, model: torch.nn.Module, val_loss: float):
        path = os.path.join(self.checkpoint_dir, "best.pt")
        torch.save({
            "val_loss":     val_loss,
            "model":        model.state_dict(),
            "run_id":       self.run_id,
            "wandb_run_id": self.run.id,      # CHANGE
        }, path)

    def finish(self):
        wandb.finish()
