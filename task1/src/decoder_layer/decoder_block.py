from src import *
from .decoder import Decoder

class DecoderBlock(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    #initialise N single decoders
    self.decoders = nn.ModuleList([Decoder(config=config) for i in range(config.n_decoder_layers)])

  def forward(self,y,encoder_input):
    out = y
    for decoder in self.decoders:
      out = decoder(out,encoder_input)
    return out
    