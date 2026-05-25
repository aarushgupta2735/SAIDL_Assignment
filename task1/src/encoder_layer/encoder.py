from src import *
from multi_encoder import MultiLayerEncoder
from feedforward import FeedForward
from add_and_norm import AddNorm

class Encoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm

  def __init__(self):
    super().__init__()
    self.ma = MultiLayerEncoder()
    self.ff = FeedForward()
    self.an1 = AddNorm()
    self.an2 = AddNorm()

  def forward(self,input):
    out1 = self.ma(input)
    out2 = self.an1(input,out1)
    out3 = self.ff(out2)
    return self.an2(out2,out3)