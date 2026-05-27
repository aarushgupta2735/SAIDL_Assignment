import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .multi_cross_decoder import MultiCrossDecoder
from .multi_self_decoder import MultiSelfDecoder
from src.feedforward import FeedForward
from src.add_and_norm import AddNorm

class Decoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm
  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.ma_cross = MultiCrossDecoder(config)
    self.ma_self = MultiSelfDecoder(config)
    self.ff = FeedForward(config)
    self.an1 = AddNorm(config)
    self.an2 = AddNorm(config)
    self.an3 = AddNorm(config)

  def forward(self,y,encoder_input):
    out1 = self.ma_self(y)
    out2 = self.an1(y,out1)
    out3 = self.ma_cross(encoder_input,out2)
    out4 = self.an2(out2,out3)
    out5 = self.ff(out4)
    return self.an3(out4,out5)                