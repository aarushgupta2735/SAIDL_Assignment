from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TransformerConfig:
    
    #Basic Configurations common to all
    vocab_size : int
    n_decoder_layers: int = 2 #6 : Reduced to prevent overfitting
    context_window: int = 1024 #T
    batch_size: int = 32 #B
    embedding_size: int = 256 #d_model C #64
    n_heads: int = 4 #8
    dropout: float = 0.2


     # --- Swappable components (drives registries) ---
    attention: str = "Standard" #Local,Sparse,MQA
    positional_encoding: str = "sinusoidal" #Rotatory,Relative,Attention  
    use_conv: bool            = False
    conv_type: str            = "none"        # "pre_attn", "interleaved", "depthwise", "gated_ffn"
 
    # --- Attention variant specific ---
    window_size: int          = 128           # sliding window attention : w - > since transformer is decoder only we will use w/2 after each token
    block_size: int           = 64            # sparse block attention
    n_kv_heads: int           = 2             # grouped query attention
 
    # --- Experiment identity ---
    experiment_name: str      = "baseline"
    run_name: Optional[str]   = None          # auto-generated from config if None

    @property
    def d_k(self):
        return self.embedding_size//self.n_heads
    
    @property
    def dff(self):
        return self.embedding_size*4
    
    def to_dict(self) -> dict:
        """Serialise config — used for wandb.init(config=...) and JSON checkpoints."""
        return asdict(self)
 
    def run_id(self) -> str:
        """
        Human-readable run name derived from config.
        e.g. 'standard_sinusoidal_T1024_L6_H4_C256'
        Used as the checkpoint folder name so each run is self-describing.
        """
        if self.run_name:
            return self.run_name
        conv_tag = f"_conv-{self.conv_type}" if self.use_conv else ""
        return (
            f"{self.attention}"
            f"_{self.positional_encoding}"
            f"{conv_tag}"
            f"_T{self.context_window}"
            f"_L{self.n_decoder_layers}"
            f"_H{self.n_heads}"
            f"_C{self.embedding_size}"
        )
