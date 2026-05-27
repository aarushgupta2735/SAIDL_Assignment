import tiktoken
import os
import torch

from src.transformer import Transformer
from config.transformer_config import TransformerConfig
from evaluation.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(config=TransformerConfig).to(device)

with open(os.path.join(os.path.dirname(__file__), "../data/wiki.val.txt"), encoding="utf-8") as f:
    val_data = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../model_checkpoint.pt")))

eval_results = evaluate(model, val_data, tokenizer, TransformerConfig, device)

print(eval_results)
