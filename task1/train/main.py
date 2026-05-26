import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

#change device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from train.loop import loop
from bpetokenizer import BPETokenizer   

with open(os.path.join(os.path.dirname(__file__), "../data/wiki.train.txt"), encoding="utf-8") as f:
    train_data = f.read()

tokenizer = BPETokenizer()
tokenizer.train(train_data, vocab_size=10000, min_frequency=2)

config = TransformerConfig(
    vocab_size = tokenizer.vocab_size  
)

print(f"vocab_size: {config.vocab_size}")
print(f"embedding_size: {config.embedding_size}")
print(f"context_window: {config.context_window}")

model = Transformer(config)
optimiser = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.98),eps=pow(10,-9))

loop(model,optimiser,train_data,TrainConfig.iterations)
