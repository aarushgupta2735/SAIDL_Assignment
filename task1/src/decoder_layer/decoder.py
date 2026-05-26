from src import *
from multi_cross_decoder import MultiCrossDecoder
from multi_self_decoder import MultiSelfDecoder
from feedforward import FeedForward
from add_and_norm import AddNorm

class Decoder(nn.Module):
  #multi-attention -- > add and norm --> feed forward --> add and norm
  
    def __init__(self):
        super().__init__()
        self.ma_cross = MultiCrossDecoder()
        self.ma_self = MultiSelfDecoder()
        self.ff = FeedForward()
        self.an1 = AddNorm()
        self.an2 = AddNorm()
        self.an3 = AddNorm()

    def forward(self,y,encoder_input):
        out1 = self.ma_self(y)
        out2 = self.an1(y,out1)
        out3 = self.ma_cross(encoder_input,out2)
        out4 = self.an2(out2,out3)
        out5 = self.ff(out4)
        return self.an3(out4,out5)              