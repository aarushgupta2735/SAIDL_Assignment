import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class SingleCrossDecoder(nn.Module):
  def __init__(self,config:TransformerConfig) -> None:
    super().__init__()
    self.t = config.d_k
    self.WQ = nn.Linear(config.embedding_size,self.t,bias=False)
    self.WK = nn.Linear(config.embedding_size,self.t,bias=False)
    self.WV = nn.Linear(config.embedding_size,self.t,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,encoder_input,decoder_input):
    Q = self.WQ(decoder_input)
    K = self.WK(encoder_input)
    V = self.WV(encoder_input)
    h = (Q@K.transpose(-2,-1))/self.t**0.5
    a = F.softmax(h,dim=-1)@V
    #dropout
    
    return self.dropModel(a)