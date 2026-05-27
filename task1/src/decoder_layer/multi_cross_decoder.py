from src import *
from .single_cross_decoder import SingleCrossDecoder

class MultiCrossDecoder(nn.Module):
    def __init__(self,config:TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SingleCrossDecoder(config) for i in range(config.n_heads)])
        self.W0 = nn.Linear(config.embedding_size,config.embedding_size)

    def forward(self,encoder_input,xt):
        heads_out = [layer(encoder_input,xt) for layer in self.layers]
        out = torch.cat(heads_out,dim=-1)
        return self.W0(out)
