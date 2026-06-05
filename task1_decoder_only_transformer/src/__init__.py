import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig