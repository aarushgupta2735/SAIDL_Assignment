from src import *

class PositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        self.T = config.context_window
        self.B = config.batch_size 
        self.C = config.embedding_size
        self.embed = nn.Embedding(config.vocab_size,config.embedding_size)
        dropModel = nn.Dropout(p=config.dropout)
    
    def forward(self,data): #adds positions encoding to (B,T,C) to give (B,T,C)
        pe = torch.zeros(self.T,self.C)
    # i is the dimenion (iteration over C) 
    # pos is the position (iteration over T)
        for pos in range(self.T):
            for i in range(self.C):
                if(i%2==0):
                    pe[pos][i] = math.sin(pos/(pow(10000,2*i/self.C)))
                else:
                    pe[pos][i] = math.cos(pos/(pow(10000,2*(i-1)/self.C)))
        #add to each batch
        for b in range(self.B):
            data[b]+=pe
