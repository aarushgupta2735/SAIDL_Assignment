import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
from .metrics import perplexity, throughput, peak_gpu_mb

#FINE
def evaluate(model, val_data, tokenizer, config:TransformerConfig, device) -> dict:
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    tokens = torch.tensor(tokenizer.encode(val_data))

    B,T = config.batch_size, config.context_window
    total_loss = 0.0
    n_batches = 0
    n_tokens = 0

    start = time.time()

    with torch.no_grad():
        for i in range(0,len(tokens)-(B*T)-1, B*T):
            xt = torch.stack([tokens[i+j*T:i+(j+1)*T] for j in range(B)], dim=0).to(device)
            yt = torch.stack([tokens[i+j*T+1:i+(j+1)*T+1] for j in range(B)], dim=0).to(device)
            _,loss= model(xt,yt)

            total_loss += loss.item()
            n_batches += 1
            n_tokens += B*T
        
        elapsed_time = time.time() - start

        avg_loss = total_loss/n_batches

        perp = perplexity(avg_loss)
        tp = throughput(n_tokens, elapsed_time)

        model.train()

        return {
            "val loss": avg_loss,
            "perplexity": perp,
            "throughput": tp,
            "peak_gpu_mb": peak_gpu_mb()
        }
    