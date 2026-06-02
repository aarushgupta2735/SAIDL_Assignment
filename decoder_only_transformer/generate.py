import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tiktoken
import platform
from src.transformer import Transformer
from config.transformer_config import TransformerConfig

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

model = Transformer(TransformerConfig(
    vocab_size = tiktoken.get_encoding("gpt2").n_vocab,
))
model.load_state_dict(torch.load(".\experiments\checkpoints\standard_sinusoidal_T1024_L2_H4_C256\ckpt_step10000.pt", map_location=device))
model = model.to(device)
model.generate_next_token("There is a cat on the roof and it is")