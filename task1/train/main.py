import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from train.loop import loop

with open(os.path.join(os.path.dirname(__file__), "../data/wiki.train.txt")) as f:
    train_data = f.read()

config = TransformerConfig()
model = Transformer(config)
optimiser = torch.optim.Adam(model.parameters(),lr=1,betas=(0.9,0.98),eps=pow(10,-9))

loop(model,optimiser,train_data,TrainConfig.iterations)
