import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig

#Might need to change
from .single_decoder.standard_single_self_decoder import StandardSingleSelfDecoder
from .single_decoder.local_single_self_decoder import LocalSingleSelfDecoder
from .single_decoder.sparse_single_self_decoder import SparseSingleSelfDecoder
from .single_decoder.mqa_single_self_decoder import MQASingleSelfDecoder

class MultiSelfDecoder(nn.Module):
    def __init__(self,config:TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([f"{config.attention}SingleSelfDecoder"(config) for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,xt):
        heads_out = [layer(xt) for layer in self.layers]
        out = torch.cat(heads_out,dim=-1)
        return self.W0(out)
