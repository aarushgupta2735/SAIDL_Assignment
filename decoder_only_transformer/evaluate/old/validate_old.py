"""
Validation loop — runs on held-out val tokens, returns avg loss.
Kept separate from training so it can be called with any model/dataset.
"""

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, val_tokens, config, num_batches: int = 50) -> float:
    """
    Runs `num_batches` random batches over val_tokens.
    Returns mean cross-entropy loss (used to compute perplexity = exp(loss)).
    """
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        ix = torch.randint(
            len(val_tokens) - config.context_window,
            (config.batch_size,),
            device=val_tokens.device,
        )
        x = torch.stack([val_tokens[j : j + config.context_window] for j in ix])
        y = torch.stack([val_tokens[j + 1 : j + 1 + config.context_window] for j in ix])

        device_type = next(model.parameters()).device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        total_loss += loss.item()

    model.train()
    return total_loss / num_batches