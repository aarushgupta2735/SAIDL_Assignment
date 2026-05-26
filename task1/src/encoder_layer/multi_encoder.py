from src import *
from single_encoder import SingleEncoder

class MultiLayerEncoder(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList( [SingleEncoder() for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,xt):
        head_out = [layer(xt) for layer in self.layers]
        out = torch.cat(head_out,dim=-1)
        return self.W0(out)