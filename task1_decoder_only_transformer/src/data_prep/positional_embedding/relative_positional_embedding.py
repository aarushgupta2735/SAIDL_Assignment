import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig



class RelativePositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.k = config.k_relative_pe
        self.d_k = config.d_k
        self.T = config.context_window
        
        self.embed = torch.nn.Embedding(2*self.k+1,self.d_k)

        # Create positional encoding ONCE in init, then register as buffer so it moves to GPU automatically
        q_pos = torch.arange(self.T).view(self.T,1)
        k_pos = torch.arange(self.T).view(1,self.T)
        relative = (q_pos-k_pos).clamp(-self.k,self.k)+self.k #(T,T)

        self.register_buffer('relative', relative)
    
    def forward(self,Q): #Q is (B,T,d_k) and R is (T,T,d_k) --> @ : (B,T,T)
        R = self.embed(self.relative)
        return torch.einsum("bid,ijd->bij",Q.float(),R.float()) 
