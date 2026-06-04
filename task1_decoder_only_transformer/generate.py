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

tokenizer = tiktoken.get_encoding("gpt2")

config = TransformerConfig(
    vocab_size          = tokenizer.n_vocab,
    attention           = "standard", #local,sparse,mqa
    positional_encoding = "rotatory", #rotatory,relative,attention 
)

model = Transformer(config).to(device)

# --- Post-training: inference latency on best checkpoint ---
print("\nLoading best checkpoint for inference latency...")
best_ckpt = torch.load(
    os.path.join("experiments", "checkpoints", config.run_id(), "best.pt"),
    map_location=device,
)
model.load_state_dict(best_ckpt["model"])
print(f"Best val loss: {best_ckpt['val_loss']:.4f}")

# --- Post-training: sample generation ---
print("\n--- Sample Generation ---")
prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
generated = prompt.clone()
model.eval()
with torch.no_grad():
    for _ in range(50):
        next_tok = model.generate_next_token(generated)
        generated = torch.cat([generated, next_tok], dim=1)
print("Tokens:", generated[0].tolist())
try:
    print("Text:", tokenizer.decode(generated[0].tolist()))
except Exception:
    print("Decoding failed (likely due to random tokens).")