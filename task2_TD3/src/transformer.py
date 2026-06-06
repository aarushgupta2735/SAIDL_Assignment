import torch
import torch.nn as nn
import torch.nn.functional as F
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task2_TD3.config.config import TD3config
from task1_decoder_only_transformer.src.decoder_layer.decoder_block import DecoderBlock

class Transformer(nn.Module):
    def __init__(self,config:TD3config, tConfig:TransformerConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.act_features+config.obs_features,tConfig.embedding_size)
        self.decoder_block = DecoderBlock(tConfig)
        self.output_projection = nn.Linear(tConfig.embedding_size,config.act_features)
        self.tanh = nn.Tanh()

    def forward(self, history_states, history_actions, pad_mask=None):
        #print(f"pad_mask: {pad_mask}")
        #print(f"history_states has nan: {torch.isnan(history_states).any()}")
        out1 = self.input_projection(torch.cat((history_actions, history_states), dim=-1))
        #print(f"after projection nan: {torch.isnan(out1).any()}")
        out1 = self.decoder_block(out1, pad_mask)
        #print(f"after decoder nan: {torch.isnan(out1).any()}")
        return self.tanh(self.output_projection(out1[:,-1,:]))        

