import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig

from task1_decoder_only_transformer.src.data_prep.positional_embedding.attention_positional_embedding import AttentionPositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.relative_positional_embedding import RelativePositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.rotatory_positional_embedding import RotatoryPositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.standard_positional_embedding import StandardPositionalEmbedding

PE ={
  "sinusoidal": StandardPositionalEmbedding,
  "rotatory": RotatoryPositionalEmbedding,
  "relative": RelativePositionalEmbedding,
  "attention": AttentionPositionalEmbedding
}


class MQASingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig,head_n):
    super().__init__()
    self.pe = config.positional_encoding
    self.pe_model = PE[self.pe](config)
    self.head_n = head_n
    self.d_k = config.d_k
    self.T = config.context_window
    self.C = config.embedding_size
    self.WQ = nn.Linear(config.embedding_size,config.d_k,bias=False) #C,d_k
    self.dropModel = nn.Dropout(p=config.dropout)
    mask = torch.triu(torch.ones(self.T, self.T), diagonal=1).bool()
    self.register_buffer('mask',mask) 

  def forward(self,xt,K,V,pad_mask=None):
    (_,T,_)= xt.shape
    Q = self.WQ(xt) #xt: (B,T,C) -> Q: (B,T,d_k)

    if(self.pe=="rotatory"):
      Q = self.pe_model(Q)
      K = self.pe_model(K)

    h = (Q@K.transpose(-2,-1))

    if(self.pe=="attention"):
      h = torch.unsqueeze(h,0)
      h=self.pe_model(h,self.head_n)
      h=torch.squeeze(h,0)
    elif(self.pe=="relative"):
      h+=self.pe_model(Q)

    h = h/self.d_k**0.5 #h: (B,T,d_k)@(B,d_k,T) -> (B,T,T)

    mask = self.mask
    if(pad_mask!=None):
      B,_ = pad_mask.shape
      p1 = torch.repeat_interleave(pad_mask,T).reshape(B,T,T)
      p2 = torch.repeat_interleave(pad_mask,T).reshape(B,T,T).transpose(-2,-1)
      p = p1|p2
      mask = self.mask|p

    h = h.masked_fill(mask[:h.shape[-2], :h.shape[-1]], float('-inf'))

    a = F.softmax(h.float(), dim=-1)
    a = torch.nan_to_num(a, nan=0.0)
    a = a.to(Q.dtype) @ V    #dropout
    #dropout
    return self.dropModel(a)
