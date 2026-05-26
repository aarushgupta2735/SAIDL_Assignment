from src import *
from encoder import Encoder

class EncoderBlock(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.model = nn.Sequential(*[Encoder() for i in range(config.n_decoder_layers)])

  def forward(self,input):
    return self.model(input)