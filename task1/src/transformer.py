from src import *
from data_prep.data_prep import DataPrep
from encoder_layer import encoder_block
from decoder_layer import decoder_block

class Transformer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size
        self.V = config.vocab_size
        #embed
        self.data_prep = DataPrep()
        #n_encoders
        self.encoder_block = encoder_block()
        #n_decoders
        self.decoder_block = decoder_block()
        #linear
        self.lin = nn.Linear(config.embedding_size,config.vocab_size)

    def forward(self,xt,yt): #takes tensors of xt,yt after build dataset
        #embed + positional encoding + encoder output + decoder output + linear + softmax
        xt = self.data_prep(xt) # N to (B,T,C)
        yt = self.data_prep(yt) # N to (B,T,C)

        encoder_output = self.encoder_block(xt) #(B,T,C)
        decoder_output = self.decoder_block(yt,encoder_output) #(B,T,C)

        logits = self.lin(decoder_output) #(b,t,v)

        loss = F.cross_entropy(logits.view(self.B*self.T,self.V),yt.view(self.B*self.T))

        return logits,loss