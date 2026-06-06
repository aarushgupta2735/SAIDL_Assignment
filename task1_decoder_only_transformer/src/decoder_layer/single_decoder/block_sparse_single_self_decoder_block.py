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

  def forward(self,xt,pad_mask=None): 
    
    Q = self.WQ(xt) #xt: (B,T,C) -> Q: (B,T,d_k) g
    K = self.WK(xt)
    V = self.WV(xt)

    if(self.pe=="rotatory"):
      Q = self.pe_model(Q)
      K = self.pe_model(K)

    w = self.w
    B,T,d_k = Q.shape

    z = T//w
    Q_chunks = Q.reshape(B,z,w,d_k) 
    K_chunks = K.reshape(B,z,w,d_k) 
    V_chunks = V.reshape(B,z,w,d_k)

    chunk_curr = Q_chunks@K_chunks.transpose(-2,-1) #(B,z,w,d_k)@(B,z,d_k,w) -> (B,z,w,w)

    if(self.pe=="attention"):
      chunk_curr = chunk_curr.reshape(z,B,w,w)
      chunk_curr = self.pe_model(chunk_curr,self.head_n)
      chunk_curr = chunk_curr.reshape(B,z,w,w)
    if(self.pe=="relative"):
      chunk_curr += self.pe_model(Q_chunks.reshape(B,T,d_k))[:, :z*w, :z*w].reshape(B,z,w,w)
      
    mask = self.mask.unsqueeze(0).unsqueeze(0)
    chunk_curr = chunk_curr.masked_fill(mask, float('-inf'))

    chunk_curr = F.softmax(chunk_curr.float(),dim=-1).to(Q.dtype)
    chunk_curr = torch.nan_to_num(chunk_curr, nan=0.0)

    chunk_curr = chunk_curr@V_chunks #(B,z,w,w) @ (B,z,w,d_k) -> (B,z,w,d_k)
    
    res = torch.reshape(chunk_curr,(B,T,d_k))

    #dropout
    return self.dropModel(res)
  
