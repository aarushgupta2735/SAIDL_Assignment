import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .multi_encoder import MultiLayerEncoder
from src.feedforward import FeedForward
from src.add_and_norm import AddNorm

class Encoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.ma = MultiLayerEncoder(config)
    self.ff = FeedForward(config)
    self.an1 = AddNorm(config)
    self.an2 = AddNorm(config)

  def forward(self,input):
    out1 = self.ma(input)
    out2 = self.an1(input,out1)
    out3 = self.ff(out2)
    return self.an2(out2,out3)