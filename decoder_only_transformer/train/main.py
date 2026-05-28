import os
import torch

#change device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from train.loop import loop,benchmark
import tiktoken

with open(os.path.join(os.path.dirname(__file__), "../data/wiki.train.txt"), encoding="utf-8") as f:
    train_data = f.read()


# Should be changed by data_prep
print("Loading tiktoken encoding...")
tokenizer = tiktoken.get_encoding("gpt2")

config = TransformerConfig(
    vocab_size = tokenizer.n_vocab
)

train_config = TrainConfig()

print(f"vocab_size: {config.vocab_size}")
print(f"embedding_size: {config.embedding_size}")
print(f"context_window: {config.context_window}")

# Check if tokenizer.encode yields a list and convert to a long tensor
print("Tokenizing the training data...")
train_tokens = tokenizer.encode(train_data)
# Convert to tensor
train_tokens = torch.tensor(train_tokens, dtype=torch.long, device=device)
print(f"Tokenization complete. Total tokens: {len(train_tokens)}")

model = Transformer(config).to(device)
optimiser = torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=pow(10,-9))

print(torch.cuda.is_available())
print(next(model.parameters()).device)
print(train_tokens.device)

benchmark(model, train_tokens, config)

loop(model, optimiser, train_tokens,config, train_config)
