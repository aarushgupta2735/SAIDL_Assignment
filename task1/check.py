import torch
from src.transformer import Transformer
from config.transformer_config import TransformerConfig  
import tiktoken

import os

# Define your target device dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Testing a Tensor ---
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Default tensor location: {x.device}")  # Output: cpu

x = x.to(device)
print(f"Moved tensor location: {x.device}")    # Output: cuda:0 (if GPU is active)

tokenizer = tiktoken.get_encoding("gpt2")
with open("./data/wiki.train.txt", encoding="utf-8") as f:
    train_data = f.read()

config = TransformerConfig(
    vocab_size = tokenizer.n_vocab  
)

# --- Testing Your Model ---
# In your project, you should do this before passing the model to the optimizer:
model = Transformer(config)
model = model.to(device) 

# Verify the model is on the GPU by checking one of its parameters)
print(f"Model location: {next(model.parameters()).device}")