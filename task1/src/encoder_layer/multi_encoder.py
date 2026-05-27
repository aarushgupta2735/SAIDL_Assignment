import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .single_encoder import SingleEncoder

class MultiLayerEncoder(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList( [SingleEncoder(config) for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,xt):
        head_out = [layer(xt) for layer in self.layers]
        out = torch.cat(head_out,dim=-1)
        return self.W0(out)