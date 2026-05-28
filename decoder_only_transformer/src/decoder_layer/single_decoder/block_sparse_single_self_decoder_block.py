import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class BlockSparseSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.d_k = config.d_k
    self.T = config.context_window
    self.C = config.embedding_size
    self.w = config.block_size
    self.WQ = nn.Linear(config.embedding_size,config.d_k,bias=False) #C,d_k
    self.WK = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.WV = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)
    #introducing sliding window of window size (w)
    mask = torch.triu(torch.ones(self.w,self.w), diagonal=1).bool()
    self.register_buffer('mask',mask) 

  def forward(self,xt):
    Q = self.WQ(xt) #xt: (B,T,C) -> Q: (B,T,d_k) g
    K = self.WK(xt)
    V = self.WV(xt)
    w = self.w
    T = self.T
    d_k = self.d_k

    Q_chunks = [Q[:,j:j+w,:].float() for j in range(0,T,w)] #Q_chunk[0] : (1,w,d_k)
    K_chunks = [K[:,j:j+w,:].float() for j in range(0,T,w)] 
    V_chunks = [V[:,j:j+w,:].float() for j in range(0,T,w)]

    chunk_curr = tuple((F.softmax((Q_chunks[i]@K_chunks[i].transpose(-2,-1)/d_k**0.5).masked_fill(self.mask, float('-inf')),dim=-1,dtype = torch.float)@V_chunks[i])[0] for i in range(len(Q_chunks)))
    res = torch.cat(chunk_curr,dim=0).nan_to_num(0).reshape(1,T,d_k)

    #dropout
    return self.dropModel(res)
