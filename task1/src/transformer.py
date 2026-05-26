from src import *
from src.data_prep.data_prep import DataPrep
from src.encoder_layer.encoder_block import EncoderBlock
from src.decoder_layer.decoder_block import DecoderBlock

class Transformer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size
        self.V = config.vocab_size
        #embed
        self.data_prep = DataPrep(config)
        #n_encoders
        self.encoder_block = EncoderBlock(config)
        #n_decoders
        self.decoder_block = DecoderBlock(config)
        #linear
        self.lin = nn.Linear(config.embedding_size,config.vocab_size)

    def forward(self,data): 
        #embed + positional encoding + encoder output + decoder output + linear + softmax
        xt_id,yt_id,xt_pe,yt_pe = self.data_prep(data) # N to (B,T) and (B,T,C)

        encoder_output = self.encoder_block(xt_pe) #(B,T,C)
        decoder_output = self.decoder_block(yt_pe,encoder_output) #(B,T,C)

        logits = self.lin(decoder_output) #(b,t,v)

        loss = F.cross_entropy(logits.view(self.B*self.T,self.V),yt_id.view(self.B*self.T))

        return logits,loss