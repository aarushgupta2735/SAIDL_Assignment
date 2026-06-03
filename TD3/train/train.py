import gymnasium as gym
import random
import torch
import gymnasium as gym
from TD3.config.config import TD3config,TransformerConfig
from src.TD3 import TD3
from src.TD3 import ReplayBuffer

def train():
    config = TD3config(
        D_size= 10**6,
        lr= 3*(10**-4),
        batch_size = 256,
        polyak_coeff = 0.005,
        policy_delay = 2,
        gamma = 0.99,
        policy_noise= 0.2,
        noise_clip = 0.5,
        exploration_noise = 0.1,
        training_iterations = 100000,
        n_envs = 3,
        BASE_SEED= 42
    )
    TransConfig = TransformerConfig(
        n_layers = 2,
        n_heads = 4,
        embed_dim = 128,
        context_window = 8 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = config.n_envs
    env_fns = [lambda i=i: gym.make("Hopper-v5", render_mode=None) for i in range(num_envs)]

    envs = gym.vector.AsyncVectorEnv(env_fns)

    seeds = [config.BASE_SEED + i for i in range(num_envs)]
    observations, infos = envs.reset(seed=seeds)
    
    buffer = ReplayBuffer(config,device)
    TD3_agent = TD3(config,TransConfig,buffer, device)

    for step in range(config.training_iterations):

        actions = TD3_agent.select_action(observations)
        next_observations, rewards, terminations, truncations, infos = envs.step(actions)
        done = 0
        if((terminations|truncations)): done = 1
        buffer.add(state=observations,action=actions,reward=rewards,next_state=next_observations,done=done)
        if(buffer.size==config.D_size):
            TD3_agent.update(step)
        observations = next_observations
        if(done):
            break

    envs.close()
