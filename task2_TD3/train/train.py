import gymnasium as gym
import numpy as np
import torch
import time
from task2_TD3.config.config import TD3config
from task2_TD3.src.TD3 import TD3, ReplayBuffer
from task2_TD3.evaluate.rl_logger import RLLogger
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig


def main():
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
        n_envs=32,
        BASE_SEED=42,
        use_transformer=False,
        exclude_x_vel=False,
        #PO
        is_velocity_hidden = True ,
        add_observation_noise =  False,
        delay_rewards = False,

    )

    tConfig = TransformerConfig(
        vocab_size=10,
        n_decoder_layers=2,
        context_window=8,
        batch_size=256,
        embedding_size=128,
        n_heads=4,
        dropout=0.1,
        pre_ln=True,
        attention="standard",
        positional_encoding="relative",
        use_conv=False,
        conv_type="none",
        window_size=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_envs = config.n_envs
    env_fns  = [lambda: gym.make("Hopper-v5", render_mode=None,
                                  exclude_current_positions_from_observation=config.exclude_x_vel)
                for _ in range(num_envs)]
    envs     = gym.vector.SyncVectorEnv(env_fns)
    eval_env = gym.make("Hopper-v5", render_mode=None,
                         exclude_current_positions_from_observation=config.exclude_x_vel)

    seeds = [config.BASE_SEED + i for i in range(num_envs)]
    obs_np, _ = envs.reset(seed=seeds)
    observations = torch.tensor(obs_np, dtype=torch.float32, device=device)

    if config.is_velocity_hidden:
        observations = torch.cat((observations[:, :5], observations[:, 7:]), dim=-1)
    if config.add_observation_noise:
        observations += torch.normal(0, config.observation_noise_std, observations.shape, device=device)

    buffer = ReplayBuffer(config, tConfig, device)
    agent  = TD3(config, tConfig, buffer, device)
    logger = RLLogger(config, run_name="td3_mlp_hopper")

    episode_rewards = np.zeros(num_envs)
    buffer_delay_rewards = np.zeros((num_envs, config.K_delayed_rewards))
    warmup_steps = tConfig.context_window * config.n_envs

    t_start = time.time()

    for step in range(config.training_iterations):
        if buffer.curr_D_size < warmup_steps:
            actions = np.stack([envs.single_action_space.sample() for _ in range(num_envs)])
        else:
            actions = agent.select_action(observations, explore=True)


        next_obs_np, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        if config.is_velocity_hidden:
            next_obs = torch.cat((next_obs[:, :5], next_obs[:, 7:]), dim=-1)
        if config.add_observation_noise:
            next_obs += torch.normal(0, config.observation_noise_std, next_obs.shape, device=device)

        if config.delay_rewards:
            if step % config.K_delayed_rewards == 0:
                rewards += buffer_delay_rewards.sum(axis=-1).astype(np.float32)
                buffer_delay_rewards = np.zeros((num_envs, config.K_delayed_rewards))
            else:
                buffer_delay_rewards[:, step % config.K_delayed_rewards] = rewards
                rewards = np.zeros(num_envs)

        real_next_obs = next_obs.clone()
        for i, done in enumerate(dones):
            if done and "final_observation" in infos and infos["final_observation"][i] is not None:
                real_next_obs[i] = torch.tensor(infos["final_observation"][i], dtype=torch.float32, device=device)

        buffer.add(
            state      = observations,
            action     = torch.tensor(actions,  dtype=torch.float32, device=device),
            reward     = torch.tensor(rewards,  dtype=torch.float32, device=device).unsqueeze(-1),
            next_state = real_next_obs,
            done       = torch.tensor(dones,    dtype=torch.float32, device=device).unsqueeze(-1),
        )

        episode_rewards += rewards
        logger.log_step(step, rewards, dones, episode_rewards, buffer.curr_D_size)
        episode_rewards[dones] = 0.0

        if buffer.curr_D_size >= config.batch_size:
            loss1, loss2, actor_loss = agent.update(step)
            logger.log_update(step=step, loss1=loss1, loss2=loss2,
                               actor_loss=actor_loss, lr=config.lr)

        if step % config.eval_iterations == 0 and step > 0:
            returns = []
            for _ in range(10):
                s, _ = eval_env.reset()
                done_eval = False
                ep_ret = 0.0
    
                while not done_eval:
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    if config.is_velocity_hidden:
                        s_t = torch.cat((s_t[:, :5], s_t[:, 7:]), dim=-1)
                    if config.add_observation_noise:
                        s_t += torch.normal(0, config.observation_noise_std, s_t.shape, device=device)

                    a = agent.select_action(s_t, explore=False, n_envs_override=1)

                    s, r, term, trunc, _ = eval_env.step(a[0])
                    ep_ret += r
                    done_eval = term or trunc
                returns.append(ep_ret)
                
            logger.log_eval(step=step, mean_return=float(np.mean(returns)),
                             std_return=float(np.std(returns)),
                             min_return=float(np.min(returns)),
                             max_return=float(np.max(returns)))

        if step % 1000 == 0 and step > 0:
            elapsed = time.time() - t_start
            print(f"Step {step} | {step/elapsed:.1f} steps/sec | buffer={buffer.curr_D_size}")

        observations = next_obs

    envs.close()
    eval_env.close()
    logger.finish()
    print("Training complete.")


if __name__ == '__main__':
    main()