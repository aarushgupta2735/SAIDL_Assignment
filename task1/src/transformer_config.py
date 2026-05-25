from dataclasses import dataclass
from src import *
from data_prep.tokensizer import Tokensizer
from bpetokenizer import BPETokenizer

@dataclass
class TransformerConfig:
    context_window: int = 1024
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    context_window: int = 1024 #T
    batch_size: int = 16 #B
    embedding_size: int =  64 #d_model C
    vocab_size : int = BPETokenizer().vocab_size
    n_heads = 8 #2
    dff: int = 4*embedding_size
    dropout = 0.1


    def d_k(self):
        return self.embedding_size//self.n_heads