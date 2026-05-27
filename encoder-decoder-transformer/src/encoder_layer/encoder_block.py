import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig

from .encoder import Encoder

class EncoderBlock(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.model = nn.Sequential(*[Encoder(config) for i in range(config.n_encoder_layers)])

  def forward(self,input):
    return self.model(input)