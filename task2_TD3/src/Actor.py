import torch
import torch.nn as nn
import torch.nn.functional as F
from task2_TD3.config.config import TD3config
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task2_TD3.src.transformer import Transformer
class Actor(nn.Module):
    def __init__(self, config: TD3config, tConfig:TransformerConfig, device):
        super().__init__()
        self.use_transformer = config.use_transformer
        if(config.use_transformer):
            self.net = Transformer(config,tConfig).to(device=device)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.obs_features, 256,bias=True),
                nn.ReLU(),
                nn.Linear(256, 256,bias=True),
                nn.ReLU(),
                nn.Linear(256, config.act_features,bias=True),
                nn.Tanh()
            ).to(device=device)
        self.a_low = config.a_low
        self.a_high = config.a_high

    def forward(self, x,history_states=None, history_actions=None, pad_mask=None): #Takes in states and gives actions
        if(self.use_transformer):
            out = self.net(history_states,history_actions,pad_mask)
        else:
            out = self.net(x)
        out = torch.clamp(out, self.a_low, self.a_high)
        return out