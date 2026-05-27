import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

class AddNorm(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.model = nn.LayerNorm(config.embedding_size)

    def forward(self, input, sublayer_output):
        return self.model(input+sublayer_output)
    