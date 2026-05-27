import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


class SingleEncoder(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.t = config.d_k
    self.WQ = nn.Linear(config.embedding_size,self.t,bias=False)
    self.WK = nn.Linear(config.embedding_size,self.t,bias=False)
    self.WV = nn.Linear(config.embedding_size,self.t,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,xt):
    Q = self.WQ(xt)
    K = self.WK(xt)
    V = self.WV(xt)
    scores = (Q@K.transpose(-2,-1))/self.t**0.5
    weights = F.softmax(scores,dim=-1)@V
    #dropout
    return self.dropModel(weights)