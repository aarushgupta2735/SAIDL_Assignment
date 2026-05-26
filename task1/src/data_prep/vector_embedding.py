from src import *

class VectorEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size,config.embedding_size)
    
    def forward(self,xt,yt): #receives (B,T) converts into B,T,C
        return self.embed(xt),self.embed(yt)