from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TransformerConfig:
    
    #Basic Configurations common to all
    vocab_size : int
    n_decoder_layers: int = 2 #6 : Reduced to prevent overfitting
    context_window: int = 1024 #T
    batch_size: int = 32 #B
    embedding_size: int = 64 #d_model C #64
    n_heads: int = 4 #8
    dropout: float = 0.3 #INCREASE IF OVERFITTING CONTINUES
    pre_ln: bool = False

    # Swappable components (drives registries) 
    attention: str = "standard" #local,sparse,mqa
    positional_encoding: str = "sinusoidal" #rotatory,relative,attention  
    use_conv: bool = False
    conv_type: str = "none"        # "pre_attn", "interleaved" ## Not implemented for now: "depthwise", "gated_ffn"
 
    #Attention variant specific
    window_size: int = 128           # sliding window attention : w - > since transformer is decoder only we will use w/2 after each token
    block_size: int  = 64            # sparse block attention
    #n_kv_heads: int   = 2             # grouped query attention -> not need explicitly
 
    #Conv Specific
    conv_pre_attn_k_size: int = 5 #kept odd as per convention and maintained in validation
    conv_interleaved_k_size: int = 5 
        #pre_attn_k_size
        ##stride_interleaved_k_size

    # Experiment identity
    experiment_name: str = "baseline"
    run_name: Optional[str] = None          # auto-generated from config if None

    @property
    def pad_pre_attn_ksize(self):
        return (self.conv_pre_attn_k_size-1)/2
    
    @property
    def pad_interleaved_k_size(self):
        return (self.conv_interleaved_k_size-1)/2
    
    @property
    def k_relative_pe(self):
        return self.context_window-1 #assumed to be T-1. Can be changed to other t-k

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

    def __post_init__(self):
        if(self.window_size>=self.context_window):
            raise ValueError(f'Window Size of sliding window attention ({self.window_size} must be less than context window ({self.context_window}))')
        if(self.embedding_size%self.n_heads!=0):
            raise ValueError(f'Embedding Size = {self.embedding_size} must be divisible by n_heads = {self.n_heads}')
        if(not(self.use_conv) and self.conv_type!="none"):
            raise ValueError(f'Conv type cannot be a valid value')
        if(not(self.conv_interleaved_k_size&1|self.conv_pre_attn_k_size&1)):
            raise ValueError(f'K_size must be odd')

