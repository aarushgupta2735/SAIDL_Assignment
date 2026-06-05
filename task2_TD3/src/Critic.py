import torch
import torch.nn as nn
import torch.nn.functional as F
from task2_TD3.config.config import TD3config


class Critic(nn.Module):
    def __init__(self, config: TD3config,device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.obs_features + config.act_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device=device)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)  
