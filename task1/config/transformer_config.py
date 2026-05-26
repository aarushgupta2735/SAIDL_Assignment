from dataclasses import dataclass
from bpetokenizer import BPETokenizer

@dataclass
class TransformerConfig:
    vocab_size : int
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    context_window: int = 1024 #T
    batch_size: int = 16 #B
    embedding_size: int =  64 #d_model C
    n_heads: int = 8 #2
    dff: int = 4*embedding_size
    dropout: int = 0.1

    @property
    def d_k(self):
        return self.embedding_size//self.n_heads
    
    
