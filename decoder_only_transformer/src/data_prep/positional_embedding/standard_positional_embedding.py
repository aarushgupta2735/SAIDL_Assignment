import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

class StandardPositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size 
        self.C = config.embedding_size
        self.dropModel = nn.Dropout(p=config.dropout)
        
        # Create positional encoding ONCE in init, then register as buffer so it moves to GPU automatically
        pe = torch.zeros(self.T, self.C)
        pos = torch.arange(0, self.T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.C, 2).float() * (-math.log(10000.0) / self.C))
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Register as buffer so it is saved with model state and moves to device along with model
        self.register_buffer('pe', pe)
    
    def forward(self,data): #adds positions encoding to (B,T,C) to give (B,T,C)

        # data is (B, T, C), self.pe is (T, C). 
        return self.dropModel(data + torch.stack([self.pe] * data.size(0))) 
