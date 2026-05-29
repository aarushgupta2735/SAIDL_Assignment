import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class LocalSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig,head_n):
    super().__init__()
    self.head_n = head_n
    self.d_k = config.d_k
    self.T = config.context_window
    self.C = config.embedding_size
    self.w = config.window_size
    self.WQ = nn.Linear(config.embedding_size,config.d_k,bias=False) #C,d_k
    self.WK = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.WV = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)
    #introducing sliding window of window size (w)
    mask_curr = torch.triu(torch.ones(self.w,self.w), diagonal=1).bool()
    mask_prev= torch.triu(torch.ones(self.w,self.w), diagonal=1).bool()
    self.register_buffer('mask_curr',mask_curr) 
    self.register_buffer('mask_prev',mask_prev) 

  def forward(self,xt):
    Q = self.WQ(xt) #xt: (B,T,C) -> Q: (B,T,d_k) g
    K = self.WK(xt)
    V = self.WV(xt)
    w = self.w
    T = self.T
    d_k = self.d_k

    Q_chunks = [Q[:,j:j+w,:].float() for j in range(0,T,w)] #Q_chunk[0] : (B,w,d_k)
    K_chunks = [K[:,j:j+w,:].float() for j in range(0,T,w)]  #K_chunks.T(-2,-1) : (B,d_k,w) 
    # @ = ( B,w,d_k) @ (B,d_k,w) -> (B,w,w) @(B,w,d_k) -> (B,w,d_k)
    V_chunks = [V[:,j:j+w,:].float() for j in range(0,T,w)]

    chunk_curr = torch.stack([(F.softmax((Q_chunks[i]@K_chunks[i].transpose(-2,-1)/d_k**0.5).masked_fill(self.mask_curr, float('-inf')),dim=-1,dtype = torch.float)@V_chunks[i]) for i in range(len(Q_chunks))])
    chunk_prev = torch.stack([(F.softmax((Q_chunks[i]@K_chunks[i-1].transpose(-2,-1)/d_k**0.5).masked_fill(~self.mask_prev,float('-inf')),dim=-1,dtype = torch.float)@V_chunks[i]) for i in range(1,len(Q_chunks))])

    chunk0 = chunk_curr[0]
    chunk_i = [chunk_curr[i+1]+chunk_prev[i] for i in range(0,(chunk_prev.shape)[0])]
    res = torch.cat((chunk0,*chunk_i),dim=1).nan_to_num(0)

    #dropout
    return self.dropModel(res)
