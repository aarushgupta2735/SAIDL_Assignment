from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size : int
    n_encoder_layers: int = 2 #6 : Reduced to prevent overfitting
    n_decoder_layers: int = 2 #6 : Reduced to prevent overfitting
    context_window: int = 1024 #T
    batch_size: int = 16 #B
    embedding_size: int =  64 #d_model C #64
    n_heads: int = 4 #8
    dropout: float = 0.1

    @property
    def d_k(self):
        return self.embedding_size//self.n_heads
    
    @property
    def dff(self):
        return self.embedding_size*4
    
    
