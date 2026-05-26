from src import *
from tokensizer import Tokensizer
from positional_embedding import PositionalEmbedding
from vector_embedding import VectorEmbedding

class DataPrep(nn.Module):
    def __init__(self):
        super().__init__()
        #only works if they all took one input + that one input was the output of the previous one: nn.Sequential(Tokensizer(), PositionalEmbedding(),VectorEmbedding())
        self.tokeniser = Tokensizer()
        self.pe = PositionalEmbedding()
        self.ve = VectorEmbedding()    

    def forward(self,data):
        xt,yt = self.tokeniser(data) #(B,T)
        xt_ve, yt_ve = self.ve(xt,yt) #(B,T,C)
        xt_pe = self.pe(xt_ve) #(B,T,C)
        yt_pe = self.pe(yt_ve) #(B,T,C)
        return xt_pe,yt_ve
        
