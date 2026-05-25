from src import *

class SingleCrossDecoder(nn.Module):
  def __init__(self,config:TransformerConfig) -> None:
    super().__init__()
    self.t = config.layer_size
    self.WQ = nn.Linear(self.t,self.t,bias=False)
    self.WK = nn.Linear(self.t,self.t,bias=False)
    self.WV = nn.Linear(self.t,self.t,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,encoder_input,decoder_input):
    Q = self.WQ(decoder_input)
    K = self.WK(encoder_input)
    V = self.WV(encoder_input)
    h = (Q@K.transpose(-2,-1))/self.t**0.5
    a = F.softmax(h,dim=-1)@V
    #dropout
    
    return self.dropModel(a)