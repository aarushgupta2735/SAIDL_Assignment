import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TD3config


class Critic(nn.Module):
    def __init__(self, config: TD3config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.obs_features + config.act_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)  
