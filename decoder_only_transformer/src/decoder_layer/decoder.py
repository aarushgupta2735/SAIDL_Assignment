import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .mqa_multi_self_decoder import MQAMultiSelfDecoder
from .multi_self_decoder import MultiSelfDecoder

from src.feedforward import FeedForward
from src.add_and_norm import AddNorm

Attention = {
    "standard": MultiSelfDecoder,
    "local": MultiSelfDecoder,
    "sparse": MultiSelfDecoder,
    "mqa": MQAMultiSelfDecoder
}

class Decoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm
  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.ma_self = Attention[config.attention](config)
    self.ff = FeedForward(config)
    self.an1 = AddNorm(config)
    self.an2 = AddNorm(config)
    self.dropout1 = nn.Dropout(p=config.dropout)
    self.dropout2 = nn.Dropout(p=config.dropout)
    
  def forward(self, x):
    x = self.an1(x + self.dropout1(self.ma_self(x)))
    x = self.an2(x + self.dropout2(self.ff(x)))
    return x