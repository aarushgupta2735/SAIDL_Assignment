from src import *
from single_self_decoder import SingleSelfDecoder

class MultiSelfDecoder(nn.Module):
    def __init__(self,config:TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SingleSelfDecoder for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,xt):
        heads_out = [layer(xt) for layer in self.layers]
        out = torch.cat(heads_out,dim=-1)
        return self.W0(out)
