import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig


class FeedForward(nn.Module):
  def __init__(self,config:TransformerConfig) -> None:
    super().__init__()
    self.W1 = nn.Linear(config.embedding_size,config.dff,bias=True)
    self.W2 = nn.Linear(config.dff,config.embedding_size,bias=True)
    self.non_linear = nn.ReLU(inplace=True)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,x):
    #FFN(x) = max(0, xW1 + b1)W2 + b2
    h1 = self.W1(x)
    h2 = self.non_linear(h1)
    h3 = self.W2(h2)
    #dropout
    return self.dropModel(h3)
