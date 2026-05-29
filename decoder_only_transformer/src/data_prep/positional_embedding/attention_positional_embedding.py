import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig



class AttentionPositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size 
        self.C = config.embedding_size
        self.dropModel = nn.Dropout(p=config.dropout)
        self.q_pos = torch.arange(self.T).view(self.T,1)
        self.k_pos = torch.arange(self.T).view(1,self.T)
        self.slopes = torch.tensor([pow(2,-8/config.n_heads)**i for i in range(1,config.n_heads+1)],dtype=float)
        self.model = self.k_pos-self.q_pos
    
    def forward(self,x,head_n): #returns the linear bias matrix based on nth head to Q.K(T) (taken as x)
        return x+(self.model*self.slopes[head_n])
