from src import *
from decoder import Decoder

class DecoderBlock(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    #initialise N single decoders
    self.decoders = nn.ModuleList([Decoder() for i in range(config.n_decoder_layers)])

  def forward(self,y,encoder_input):
    for i in range(len(self.decoders)):
        if(i==0):
            out = self.decoders[i](y,encoder_input)
        else:
            out = self.decoders[i](out,encoder_input)
    