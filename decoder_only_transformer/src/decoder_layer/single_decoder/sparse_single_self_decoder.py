import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class SparseSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.d_k = config.d_k
    self.T = config.context_window
    self.C = config.embedding_size
    self.WQ = nn.Linear(config.embedding_size,config.d_k,bias=False) #C,d_k
    self.WK = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.WV = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)
    mask = torch.triu(torch.ones(self.T, self.T), diagonal=1).bool()
    self.register_buffer('mask',mask) 

  def forward(self,xt):
    Q = self.WQ(xt) #xt: (B,T,C) -> Q: (B,T,d_k)
    K = self.WK(xt)
    V = self.WV(xt)
    h = (Q@K.transpose(-2,-1))/self.d_k**0.5 #h: (B,T,d_k)@(B,d_k,T) -> (B,T,T)

    # Ensure mask inherits the GPU device from the h tensor
    
    h = h.masked_fill(self.mask, float('-inf'))

    a = F.softmax(h,dim=-1)@V
    #dropout
    return self.dropModel(a)
