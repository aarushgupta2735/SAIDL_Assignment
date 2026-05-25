from src import *

class SingleEncoder(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.t = config.layer_size
    self.WQ = nn.Linear(self.t,self.t,bias=False)
    self.WK = nn.Linear(self.t,self.t,bias=False)
    self.WV = nn.Linear(self.t,self.t,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,xt):
    Q = self.WQ(xt)
    K = self.WK(xt)
    V = self.WV(xt)
    scores = (Q@K.transpose(-2,-1))/self.t**0.5
    weights = F.softmax(scores,dim=-1)@V
    #dropout
    return self.dropModel(weights)