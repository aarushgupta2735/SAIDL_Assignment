import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig
from task1_decoder_only_transformer.src.layerNorm import AddNorm
from task1_decoder_only_transformer.src.decoder_layer.decoder import Decoder

pre_ln = {
  True: AddNorm,
  False: nn.Identity
}

class DecoderBlock(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    #initialise N single decoders
    self.decoders = nn.ModuleList([Decoder(config=config) for i in range(config.n_decoder_layers)])
    self.an = pre_ln[config.pre_ln](config)

  def forward(self,input,pad_mask=None):
    for decoder in self.decoders:
      input = decoder(input,pad_mask)
    return self.an(input)
    