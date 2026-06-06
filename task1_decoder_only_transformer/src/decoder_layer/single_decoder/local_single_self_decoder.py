import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig

from task1_decoder_only_transformer.src.data_prep.positional_embedding.attention_positional_embedding import AttentionPositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.relative_positional_embedding import RelativePositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.rotatory_positional_embedding import RotatoryPositionalEmbedding
from task1_decoder_only_transformer.src.data_prep.positional_embedding.standard_positional_embedding import StandardPositionalEmbedding

PE ={
  "sinusoidal": StandardPositionalEmbedding,
  "rotatory": RotatoryPositionalEmbedding,
  "relative": RelativePositionalEmbedding,
  "attention": AttentionPositionalEmbedding
}


class LocalSingleSelfDecoder(nn.Module):
  #masking
  def __init__(self,config:TransformerConfig,head_n):
    super().__init__()
    self.pe = config.positional_encoding
    self.pe_model = PE[self.pe](config)
    self.head_n = head_n
    self.d_k = config.d_k
    self.T = config.context_window
    self.C = config.embedding_size
    self.w = config.window_size
    self.WQ = nn.Linear(config.embedding_size,config.d_k,bias=False) #C,d_k
    self.WK = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.WV = nn.Linear(config.embedding_size,config.d_k,bias=False)
    self.dropModel = nn.Dropout(p=config.dropout)
    #introducing sliding window of window size (w)
    mask_curr = torch.triu(torch.ones(self.w,self.w), diagonal=1).bool()

    #mask_prev= torch.triu(torch.ones(self.w,self.w), diagonal=1).bool()
    self.register_buffer('mask_curr',mask_curr) 

def forward(self, xt, pad_mask=None):
    Q = self.WQ(xt)
    K = self.WK(xt)
    V = self.WV(xt)

    w = self.w
    B, T, d_k = Q.shape
    z = T // w

    if self.pe == "rotatory":
        Q = self.pe_model(Q)
        K = self.pe_model(K)

    Q_chunks = [Q[:, j:j+w, :] for j in range(0, T, w)]
    K_chunks = [K[:, j:j+w, :] for j in range(0, T, w)]
    V_chunks = [V[:, j:j+w, :] for j in range(0, T, w)]

    # --- chunk_curr: each chunk attends to itself ---
    # step 1: scores
    chunk_curr = torch.stack([Q_chunks[i] @ K_chunks[i].transpose(-2,-1) 
                               for i in range(z)], dim=1)  # (B,z,w,w)

    # step 2: PE bias
    if self.pe == "attention":
        chunk_curr = self.pe_model(chunk_curr, self.head_n)
    if self.pe == "relative":
        chunk_curr += self.pe_model(Q).reshape(B, z, w, w)

    # step 3: causal mask
    chunk_curr = chunk_curr.masked_fill(
        self.mask_curr.unsqueeze(0).unsqueeze(0), float('-inf')
    )

    # step 4: softmax
    chunk_curr = F.softmax(chunk_curr.float(), dim=-1).to(Q.dtype)
    chunk_curr = torch.nan_to_num(chunk_curr, nan=0.0)

    # step 5: multiply V
    V_curr = torch.stack(V_chunks, dim=1)       # (B,z,w,d_k)
    chunk_curr = chunk_curr @ V_curr            # (B,z,w,d_k)

    # --- chunk_prev: each chunk attends to previous chunk ---
    # step 1: scores
    chunk_prev = torch.stack([Q_chunks[i] @ K_chunks[i-1].transpose(-2,-1) 
                               for i in range(1, z)], dim=1)  # (B,z-1,w,w)

    # step 2: PE bias
    if self.pe == "attention":
        chunk_prev = self.pe_model(chunk_prev, self.head_n)
    if self.pe == "relative":
        chunk_prev += self.pe_model(Q[:, w:, :]).reshape(B, z-1, w, w)

    # step 3: no causal mask for prev chunk — all previous tokens are in the past

    # step 4: softmax
    chunk_prev = F.softmax(chunk_prev.float(), dim=-1).to(Q.dtype)
    chunk_prev = torch.nan_to_num(chunk_prev, nan=0.0)

    # step 5: multiply V
    V_prev = torch.stack(V_chunks[1:], dim=1)   # (B,z-1,w,d_k)
    chunk_prev = chunk_prev @ V_prev            # (B,z-1,w,d_k)

    # --- combine ---
    chunk0 = chunk_curr[:, 0]                   # (B,w,d_k)
    chunk_i = [chunk_curr[:, i+1] + chunk_prev[:, i] 
               for i in range(chunk_prev.shape[1])]
    res = torch.cat([chunk0] + chunk_i, dim=1)  # (B,T,d_k)

    return self.dropModel(res)