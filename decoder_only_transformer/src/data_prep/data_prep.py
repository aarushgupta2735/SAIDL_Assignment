import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .positional_embedding import PositionalEmbedding
from .vector_embedding import VectorEmbedding

class DataPrep(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pe = PositionalEmbedding(config)
        self.ve = VectorEmbedding(config)    

    def forward(self, xt_id, yt_id):
        xt_ve, yt_ve = self.ve(xt_id,yt_id) #(B,T,C)
        xt_pe = self.pe(xt_ve) #(B,T,C)
        yt_pe = self.pe(yt_ve) #(B,T,C)
        return xt_id,yt_id,xt_pe,yt_pe #(B,T) and #(B,T,C) 
         
