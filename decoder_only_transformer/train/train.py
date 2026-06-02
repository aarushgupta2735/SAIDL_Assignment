import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tiktoken
import platform
from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from .loop import loop
from evaluate.logger import ExperimentLogger

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# --- Tokenizer ---
print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")

# --- Data ---
print("Loading data...")
data_dir = os.path.join(os.path.dirname(__file__), "../data")

with open(os.path.join(data_dir, "wiki.train.txt"), encoding="utf-8") as f:
    train_data = f.read()
with open(os.path.join(data_dir, "wiki.valid.txt"), encoding="utf-8") as f:
    val_data = f.read()

print("Tokenizing...")
train_tokens = torch.tensor(tokenizer.encode(train_data), dtype=torch.long, device=device)
val_tokens   = torch.tensor(tokenizer.encode(val_data),   dtype=torch.long, device=device)
print(f"Train tokens: {len(train_tokens):,}  |  Val tokens: {len(val_tokens):,}")

# --- Config ---
config = TransformerConfig(
    vocab_size        = tokenizer.n_vocab,
    experiment_name   = "baseline",      # change per experiment group
    attention         = "standard",
    positional_encoding = "sinusoidal",
)
train_config = TrainConfig()

# --- Model ---
model = Transformer(config).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# --- Optimiser ---
optimiser = torch.optim.Adam(
    model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    # lr=1.0 because update_lr overrides it every step
)

# --- Logger ---
logger = ExperimentLogger(config, train_config)

# --- Train ---
print(f"Starting run: {config.run_id()}")
loop(model, optimiser, train_tokens, val_tokens, config, train_config, logger)

# --- Post-training: inference latency ---
logger.log_inference_latency(model, device)

logger.finish()
print("Done.")