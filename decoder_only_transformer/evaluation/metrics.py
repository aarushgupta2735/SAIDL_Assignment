import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


def perplexity(loss):
    return math.exp(loss)

def throughput(num_tokens, time_seconds):
    return num_tokens / time_seconds

def peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)