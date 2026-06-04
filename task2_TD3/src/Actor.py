import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TD3config

class Actor(nn.Module):
    def __init__(self, config: TD3config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config.act_features),
            nn.Tanh()
        )
        self.a_low = config.a_low
        self.a_high = config.a_high

    def forward(self, x):
        x = self.net(x)
        x = torch.clip(x, self.a_low, self.a_high)
        return x