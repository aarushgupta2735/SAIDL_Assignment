from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class TrainConfig():
    warmup_steps: int = 4000
    iterations: int = 10000
    val_interval: int = 500     # run validation every N steps

    
    
    