import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TD3config, TransformerConfig
from task1_decoder_only_transformer.src.transformer import Transformer
class Actor(nn.Module):
    def __init__(self, config: TD3config, TransConfig:TransformerConfig, device):
        super().__init__()
        if(config.use_transformer):
            self.net = Transformer(TransformerConfig=TransConfig).to(device=device)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.obs_features, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, config.act_features),
                nn.Tanh()
            ).to(device=device)
        self.a_low = config.a_low
        self.a_high = config.a_high

    def forward(self, x): #Takes in states and gives actions

        x = self.net(x)
        x = torch.clip(x, self.a_low, self.a_high)
        return x