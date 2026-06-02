import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from src.data_prep.positional_embedding.attention_positional_embedding import AttentionPositionalEmbedding
from src.data_prep.positional_embedding.relative_positional_embedding import RelativePositionalEmbedding
from src.data_prep.positional_embedding.rotatory_positional_embedding import RotatoryPositionalEmbedding
from src.data_prep.positional_embedding.standard_positional_embedding import StandardPositionalEmbedding

PE ={
  "sinusoidal": StandardPositionalEmbedding,
  "rotatory": RotatoryPositionalEmbedding,
  "relative": RelativePositionalEmbedding,
  "attention": AttentionPositionalEmbedding
}

class StandardSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig,head_n):
    super().__init__()
    self.head_n = head_n
    self.d_k = config.d_k
    self.pe = config.positional_encoding
    self.pe_model = PE[self.pe](config)

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

    if(self.pe=="rotatory"):
      Q = self.pe_model(Q)
      K = self.pe_model(K)
    
    h = (Q@K.transpose(-2,-1)) #h: (B,T,d_k)@(B,d_k,T) -> (B,T,T)

    if(self.pe=="attention"):
      h = torch.unsqueeze(h,0)
      h=self.pe_model(h,self.head_n)
      h=torch.squeeze(h,0)
    if(self.pe=="relative"):
      h+=self.pe_model(Q)
    
    h = h/self.d_k**0.5

    # Ensure mask inherits the GPU device from the h tensor
    
    h = h.masked_fill(self.mask, float('-inf'))

    a = F.softmax(h,dim=-1)@V
    #dropout
    return self.dropModel(a)
