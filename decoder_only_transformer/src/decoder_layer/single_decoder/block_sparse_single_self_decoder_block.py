import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from data_prep.positional_embedding.attention_positional_embedding import AttentionPositionalEmbedding
from data_prep.positional_embedding.relative_positional_embedding import RelativePositionalEmbedding
from data_prep.positional_embedding.rotatory_positional_embedding import RotatoryPositionalEmbedding
from data_prep.positional_embedding.standard_positional_embedding import StandardPositionalEmbedding

PE ={
  "Standard": StandardPositionalEmbedding,
  "Rotatory": RotatoryPositionalEmbedding,
  "Relative": RelativePositionalEmbedding,
  "Attention ": AttentionPositionalEmbedding
}


class BlockSparseSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig,head_n):
    super().__init__()
    self.head_n = head_n
    self.pe = config.positional_encoding
    self.pe_model = PE[self.pe](config)
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

    if(self.pe=="Rotatory"):
      Q = self.pe_model(Q)
      K = self.pe_model(K)

    w = self.w
    T = self.T
    d_k = self.d_k

    Q_chunks = [Q[:,j:j+w,:].float() for j in range(0,T,w)] #Q_chunk[0] : (B,w,d_k)
    K_chunks = [K[:,j:j+w,:].float() for j in range(0,T,w)]  #K_chunks.T(-2,-1) : (B,d_k,w) 
    # @ = ( B,w,d_k) @ (B,d_k,w) -> (B,w,w) @(B,w,d_k) -> (B,w,d_k)
    V_chunks = [V[:,j:j+w,:].float() for j in range(0,T,w)]

    chunk_curr = tuple([(F.softmax((Q_chunks[i]@K_chunks[i].transpose(-2,-1)/d_k**0.5).masked_fill(self.mask, float('-inf')),dim=-1,dtype = torch.float)@V_chunks[i]) for i in range(len(Q_chunks))])
    
    #YET TO INTEGRATE 

    if(self.pe=="Attention"):
      h=self.pe_model(h,self.head_n)
    if(self.pe=="Relative"):
      h+=self.pe_model(Q)
    
    res = torch.cat(chunk_curr,dim=1).nan_to_num(0) 

    #dropout
    return self.dropModel(res)
