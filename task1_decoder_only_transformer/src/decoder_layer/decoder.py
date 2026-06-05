import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from  task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig

from task1_decoder_only_transformer.src.decoder_layer.mqa_multi_self_decoder import MQAMultiSelfDecoder
from task1_decoder_only_transformer.src.decoder_layer.multi_self_decoder import MultiSelfDecoder

from task1_decoder_only_transformer.src.feedforward import FeedForward
from task1_decoder_only_transformer.src.layerNorm import AddNorm

Attention = {
    "standard": MultiSelfDecoder,
    "local": MultiSelfDecoder,
    "sparse": MultiSelfDecoder,
    "mqa": MQAMultiSelfDecoder
}

ConvArch = {
    "none": [nn.Identity, nn.Identity],
    "pre_attn": [nn.Conv1d, nn.Identity],
    "interleaved" : [nn.Conv1d, nn.Conv1d]
}

class Decoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm
  def __init__(self, config: TransformerConfig):
      super().__init__()
      self.C = config.embedding_size
      self.conv1 = ConvArch[config.conv_type][0](self.C, self.C, config.conv_pre_attn_k_size, 1, config.pad_pre_attn_ksize)
      self.conv2 = ConvArch[config.conv_type][1](self.C, self.C, config.conv_interleaved_k_size, 1, config.pad_interleaved_k_size)
      self.ma_self = Attention[config.attention](config)
      self.ff = FeedForward(config)
      self.an1 = AddNorm(config)
      self.an2 = AddNorm(config)
      self.pre_ln = config.pre_ln
      self.dropout1 = nn.Dropout(p=config.dropout)
      self.dropout2 = nn.Dropout(p=config.dropout)

  def forward(self, x, pad_mask=None):  # (B,T,C)
      if self.pre_ln:
          x = x + self.dropout1(self.ma_self(self.an1(self.conv1(x.transpose(-2,-1)).transpose(-2,-1)), pad_mask))
          x = x + self.dropout1(self.conv2(x.transpose(-2,-1)).transpose(-2,-1))  # interleaved conv after attention
          x = x + self.dropout2(self.ff(self.an2(x)))
      else:
          x = self.an1(x + self.dropout1(self.ma_self(self.conv1(x.transpose(-2,-1)).transpose(-2,-1), pad_mask)))
          x = self.an2(x + self.dropout1(self.conv2(x.transpose(-2,-1)).transpose(-2,-1)))
          x = self.an2(x + self.dropout2(self.ff(x)))
      return x