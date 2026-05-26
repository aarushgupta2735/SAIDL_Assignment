from src import *

class SingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.T = config.d_k
    self.WQ = nn.Linear(config.d_k,config.d_k,bias=False)
    self.WK = nn.Linear(config.d_k,config.d_k,bias=False)
    self.WV = nn.Linear(config.d_k,config.d_k,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)

  def forward(self,xt):
    Q = self.WQ(xt)
    K = self.WK(xt)
    V = self.WV(xt)
    h = (Q@K.transpose(-2,-1))/self.T**0.5

    mask = torch.triu(torch.ones(self.T, self.T), diagonal=1).bool() 
    h = h.masked_fill(mask, float('-inf'))

    a = F.softmax(h,dim=-1)@V
    #dropout
    return self.dropModel(a)
