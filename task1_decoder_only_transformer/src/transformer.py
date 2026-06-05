import torch
import torch.nn as nn
import torch.nn.functional as F

from config.transformer_config import TransformerConfig

from task1_decoder_only_transformer.src.data_prep.data_prep import DataPrep
from task1_decoder_only_transformer.src.decoder_layer.decoder_block import DecoderBlock
from task1_decoder_only_transformer.src.data_prep.positional_embedding.standard_positional_embedding import StandardPositionalEmbedding


class Transformer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        
        self.T = config.context_window
        self.B = config.batch_size
        self.V = config.vocab_size
        #embed
        self.data_prep = DataPrep(config)
        #n_decoders
        self.decoder_block = DecoderBlock(config)
        #linear
        self.lin = nn.Linear(config.embedding_size,config.vocab_size)
        self.lin.weight = self.data_prep.ve.embed.weight

    def forward(self, x, y): 
        #embed + positional encoding + encoder output + decoder output + linear + softmax
        xt_pe = self.data_prep(x) # (B,T) to (B,T,C) 
        decoder_output = self.decoder_block(xt_pe) #(B,T,C)
        logits = self.lin(decoder_output) #(b,t,v)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T,V),y.view(B*T))

        return logits,loss

    @torch.no_grad()
    def generate_next_token(self,seq): #seq : (1,T) ##Generate the next token 
        xt_pe = self.data_prep(seq[:,-self.T:]) # (1,T) to (1,T,C) 
        decoder_output = self.decoder_block(xt_pe) #(1,T,C)
        logits = self.lin(decoder_output) #(1,T,V)
        req_token_index = torch.argmax(logits[:,-1,:],dim=-1).unsqueeze(0)
        return req_token_index
