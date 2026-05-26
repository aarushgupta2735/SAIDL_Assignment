import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from train.loop import loop
from bpetokenizer import BPETokenizer   

with open(os.path.join(os.path.dirname(__file__), "../data/wiki.train.txt"), encoding="utf-8") as f:
    train_data = f.read()

tokenizer = BPETokenizer()
tokenizer.train(train_data)   
config = TransformerConfig(
    vocab_size = tokenizer.vocab_size  
)

print(f"vocab_size: {TransformerConfig.vocab_size}")
print(f"embedding_size: {TransformerConfig.embedding_size}")
print(f"context_window: {TransformerConfig.context_window}")

model = Transformer(config)
optimiser = torch.optim.Adam(model.parameters(),lr=1,betas=(0.9,0.98),eps=pow(10,-9))

loop(model,optimiser,train_data,TrainConfig.iterations)
