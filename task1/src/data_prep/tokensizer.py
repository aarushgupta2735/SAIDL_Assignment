from src import *
from bpetokenizer import BPETokenizer

#prepares the data into a (B,T,C) matrix 
class Tokensizer(nn.Module):
  def __init__(self,config:TransformerConfig):
    super().__init__()
    self.tokenizer = BPETokenizer()
    self.T = config.context_window
    self.B = config.batch_size 
    self.C = config.embedding_size

  #tokenizes the entire data and calculates vocab_size
  def forward(self,data):
    #tokenise and slice into (xt,yt) pairs
    t = self.tokenizer.encode(data)
    out_x = []
    out_y = []
    for i in range(self.B):
      if((i+1)*self.T>len(t)): break;
      out_x.append(t[i*self.T:(i+1)*self.T])
      out_y.append(t[1+i*self.T:1+(i+1)*self.T])
    xt = torch.tensor(out_x)
    yt=  torch.tensor(out_y)
    


  