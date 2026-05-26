from dataclasses import dataclass
from src import *
from src.data_prep.tokensizer import Tokensizer
from bpetokenizer import BPETokenizer

@dataclass
class TrainConfig():
    step_num: int = 1
    warmup_steps: int = 4000
    iterations: int = 1000

    
    
