import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig

class RotatoryPositionalEmbedding(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.T = config.context_window
        self.B = config.batch_size 
        self.C = config.embedding_size
        self.d_k = config.d_k
        
        # Create positional encoding ONCE in init, then register as buffer so it moves to GPU automatically
        theta = torch.tensor([pow(10000,(-2*t)/self.d_k) for t in range(self.d_k//2)]) #(d_k//2)
        m = torch.arange(self.T).unsqueeze(1) #(T,1)
        m_theta = theta*m #(T,d_k//2)
        cos_theta = torch.cos(m_theta).repeat_interleave(2,dim=-1)  #(T,d_k) by repeating each value twice for interleaving
        sin_theta = torch.sin(m_theta).repeat_interleave(2,dim=-1)

        self.register_buffer('cos_theta', cos_theta)
        self.register_buffer('sin_theta', sin_theta)

    
    def forward(self,x): #input is (B,T,d_k) to give (B,T,d_k)
        #check if this is correct on interference
        B,T,d_k = x.shape
        x = torch.reshape(x,(B,T,d_k//2,2))
        x1 = x[:,:,:,0]
        x2  = x[:,:,:,1]
        x = x.reshape(B,T,d_k)
        res_x = torch.stack((-x2,x1),dim=-1).reshape(B,T,d_k)

        return res_x*self.sin_theta[:T,:] + x*self.cos_theta[:T,:]
    
