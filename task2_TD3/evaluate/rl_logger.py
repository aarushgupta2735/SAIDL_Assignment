"""
RL Experiment Logger — tracks all metrics required by the SAiDL Core RL assignment.
"""
import time
import numpy as np
import torch
import wandb
from dataclasses import asdict


class RLLogger:
    def __init__(self, config, run_name: str = "td3_mlp"):
        self.config = config
        self.run = wandb.init(
            project="saidl-rl",
            name=run_name,
            config=asdict(config),
        )
        self.episode_count = 0
        self.step_start = time.perf_counter()
        self._last_step_time = time.perf_counter()

    def log_step(self, step: int, rewards: np.ndarray, dones: np.ndarray,
                 episode_rewards: torch.Tensor, buffer_size: int):
        """Call every environment step."""
        now = time.perf_counter()
        steps_per_sec = 1.0 / max(now - self._last_step_time, 1e-9)
        self._last_step_time = now

        metrics = {
            "step": step,
            "perf/steps_per_sec": steps_per_sec,
            "buffer/size": buffer_size,
            "reward/step_mean": float(np.mean(rewards)),
        }

        # Log completed episodes
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                metrics[f"reward/episode_return"] = float(episode_rewards[i])
                metrics[f"episodes/count"] = self.episode_count

        wandb.log(metrics, step=step)

    def log_update(self, step: int, loss1: float, loss2: float,
                   actor_loss: float = None, lr: float = None,
                   q1_mean: float = None, q2_mean: float = None):
        """Call every TD3 update step."""
        metrics = {
            "train/critic1_loss": loss1,
            "train/critic2_loss": loss2,
            "train/total_critic_loss": loss1 + loss2,
        }
        if actor_loss is not None:
            metrics["train/actor_loss"] = actor_loss
        if lr is not None:
            metrics["train/lr"] = lr
        if q1_mean is not None:
            metrics["train/q1_mean"] = q1_mean
            metrics["train/q2_mean"] = q2_mean

        wandb.log(metrics, step=step)

    def log_eval(self, step: int, mean_return: float, std_return: float,
                 min_return: float, max_return: float, n_episodes: int = 10):
        """Call after deterministic evaluation rollouts."""
        wandb.log({
            "eval/mean_return":   mean_return,
            "eval/std_return":    std_return,
            "eval/min_return":    min_return,
            "eval/max_return":    max_return,
            "eval/n_episodes":    n_episodes,
        }, step=step)
        print(
            f"[Eval @ {step}] mean={mean_return:.1f} "
            f"std={std_return:.1f} min={min_return:.1f} max={max_return:.1f}"
        )

    def log_attention(self, step: int, attn_weights: torch.Tensor, entropy: float):
        """For Transformer-TD3: log attention weights and entropy."""
        wandb.log({
            "attention/entropy": entropy,
            "attention/mean_weight_t-1": float(attn_weights[..., -1].mean()),
            "attention/mean_weight_t-4": float(attn_weights[..., -4].mean()) if attn_weights.shape[-1] >= 4 else 0,
        }, step=step)

    def finish(self):
        wandb.finish()