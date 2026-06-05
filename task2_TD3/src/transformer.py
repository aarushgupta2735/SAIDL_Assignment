import torch
import torch.nn as nn
import torch.nn.functional as F
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from config.config import TD3config
from task1_decoder_only_transformer.src.decoder_layer.decoder_block import DecoderBlock

class Transformer(nn.Module):
    def __init__(self,config:TD3config, tConfig:TransformerConfig):
        super().__init__()
        self.input_projection = nn.Linear((config.act_features,config.obs_features),tConfig.embed_dim)
        self.decoder_block = nn.ModuleList([DecoderBlock(tConfig) for i in range(tConfig.n_layers)])
        self.output_projection = nn.Linear((tConfig.embed_dim,config.act_features))
        self.tanh = nn.Tanh()

    def forward(self,history_states,history_actions): #(L,obs_features+act_features) 
        out1 = self.input_projection(torch.cat((history_actions,history_states),dim=1))
        for decoder in self.decoder_block:
            out1 = decoder(out1)
        return self.tanh(self.output_projection(out1)) #returns action between [-1,1]
        

