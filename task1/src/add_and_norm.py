from src import *
class AddNorm(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.model = nn.LayerNorm(config.embedding_size)

    def forward(self, input, sublayer_output):
        return self.model(input+sublayer_output)
    