from src import *
from tokensizer import Tokensizer
from positional_embedding import PositionalEmbedding
from vector_embedding import VectorEmbedding

class DataPrep(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential([Tokensizer(), PositionalEmbedding(),VectorEmbedding()])
    
    def forward(self,data):
        return self.model(data)