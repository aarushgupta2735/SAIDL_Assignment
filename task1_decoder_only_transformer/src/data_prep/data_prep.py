import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .positional_embedding.standard_positional_embedding import StandardPositionalEmbedding

PE = {
    "sinusoidal": StandardPositionalEmbedding,
    "rotatory": nn.Identity,
    "relative": nn.Identity,
    "attention": nn.Identity,
}
from .vector_embedding import VectorEmbedding

class DataPrep(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pe = PE[config.positional_encoding](config)
        self.ve = VectorEmbedding(config)    

    def forward(self, xt_id): #Input: (B,T)
        xt_ve = self.ve(xt_id) #(B,T,C)
        xt_pe = self.pe(xt_ve) #(B,T,C)
        return xt_pe #(B,T,C) 
         
