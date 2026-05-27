from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class TrainConfig():
    step_num: int = 1
    warmup_steps: int = 400
    iterations: int = 10000

    
    
