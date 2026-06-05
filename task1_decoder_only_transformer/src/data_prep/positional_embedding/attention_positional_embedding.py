import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig



class AttentionPositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size 
        self.C = config.embedding_size
        self.dropModel = nn.Dropout(p=config.dropout)
        self.register_buffer('slopes',torch.tensor([pow(2,-8/config.n_heads)**i for i in range(1,config.n_heads+1)]).to(dtype=torch.float32))
        self.register_buffer('model',(torch.arange(self.T).view(self.T,1)-torch.arange(self.T).view(1,self.T)).to(torch.float32))
    
    def forward(self,x,head_n): #returns the linear bias matrix based on nth head to Q.K(T) (taken as x)
        return x + (self.model[:x.shape[-2], :x.shape[-1]] * self.slopes[head_n]).to(x.dtype)
