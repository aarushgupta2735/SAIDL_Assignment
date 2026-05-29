import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig

from .single_decoder.mqa_single_self_decoder import MQASingleSelfDecoder



class MQAMultiSelfDecoder(nn.Module):
    def __init__(self,config:TransformerConfig) -> None:
        super().__init__()
        self.WK = nn.Linear(config.embedding_size,config.d_k,bias=False)
        self.WV = nn.Linear(config.embedding_size,config.d_k,bias=False)
        self.layers = nn.ModuleList([MQASingleSelfDecoder(config) for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,xt):
        K = self.WK(xt)
        V = self.WV(xt)
        heads_out = [layer(xt,K,V) for layer in self.layers]
        out = torch.cat(heads_out,dim=-1)
        return self.W0(out)
