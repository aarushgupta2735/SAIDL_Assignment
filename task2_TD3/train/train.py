import gymnasium as gym
import numpy as np
import torch
import wandb

from config.config import TD3config
from src.TD3 import TD3, ReplayBuffer
from evaluate.rl_logger import RLLogger
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    config = TD3config(
        D_size=10**6,
        lr=3e-4,
        batch_size=256,
        polyak_coeff=0.005,
        policy_delay=2,
        gamma=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
        training_iterations=1_000_000,
        n_envs=3,
        BASE_SEED=42,
    )

    tConfig = TransformerConfig(
        vocab_size = 10,
        n_decoder_layers = 2,
        context_window = 8, #L
        batch_size = 256, #B
        embedding_size = 128, #d_model C #64
        n_heads = 4, #8
        dropout = 0.1 #INCREASE IF OVERFITTING CONTINUES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Environments ──────────────────────────────────────────────────────────
    num_envs = config.n_envs
    env_fns  = [lambda: gym.make("Hopper-v5", render_mode=None) for _ in range(num_envs)]
    envs     = gym.vector.AsyncVectorEnv(env_fns)

    eval_env = gym.make("Hopper-v5", render_mode=None)

    seeds = [config.BASE_SEED + i for i in range(num_envs)]
    obs_np, _ = envs.reset(seed=seeds)
    observations = torch.tensor(obs_np, dtype=torch.float32, device=device)

    # ── Agent & Logger ────────────────────────────────────────────────────────
    buffer    = ReplayBuffer(config,tConfig, device)
    agent     = TD3(config, tConfig, buffer, device)
    logger    = RLLogger(config, run_name="td3_mlp_hopper")

    episode_rewards = np.zeros(num_envs)

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(config.training_iterations):

        actions = agent.select_action(observations, explore=True)

        next_obs_np, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        real_next_obs = next_obs_np.copy()
        for i, done in enumerate(dones):
            if done and "final_observation" in infos and infos["final_observation"][i] is not None:
                real_next_obs[i] = infos["final_observation"][i]

        buffer.add(
            state      = observations,
            action     = torch.tensor(actions,       dtype=torch.float32, device=device),
            reward     = torch.tensor(rewards,       dtype=torch.float32, device=device).unsqueeze(-1),
            next_state = torch.tensor(real_next_obs, dtype=torch.float32, device=device),
            done       = torch.tensor(dones,         dtype=torch.float32, device=device).unsqueeze(-1),
        )

        episode_rewards += rewards
        logger.log_step(step, rewards, dones, episode_rewards, buffer.curr_D_size)
        episode_rewards[dones] = 0.0

        if buffer.curr_D_size >= config.batch_size:
            loss1, loss2, actor_loss = agent.update(step)
            if loss1 is not None:
                logger.log_update(
                    step=step,
                    loss1=loss1,
                    loss2=loss2,
                    actor_loss=actor_loss,
                    lr=config.lr,
                )

        if step % config.eval_iterations == 0 and step > 0:
            returns = []
            for _ in range(10):
                s, _ = eval_env.reset()
                done_eval = False
                ep_ret = 0.0
                while not done_eval:
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a = agent.select_action(s_t, explore=False)
                    s, r, term, trunc, _ = eval_env.step(a[0])
                    ep_ret += r
                    done_eval = term or trunc
                returns.append(ep_ret)
            logger.log_eval(
                step=step,
                mean_return=float(np.mean(returns)),
                std_return=float(np.std(returns)),
                min_return=float(np.min(returns)),
                max_return=float(np.max(returns)),
            )

        observations = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    envs.close()
    eval_env.close()
    logger.finish()
    print("Training complete.")


if __name__ == '__main__':
    main()