"""
Validation loop — runs on held-out val tokens, returns avg loss.
Kept separate from training so it can be called with any model/dataset.
"""

import torch


@torch.no_grad()
# CHANGE: device parameter added so autocast uses correct device_type
def evaluate(model, val_tokens, config, device: torch.device) -> float:
    """
    Iterates sequentially through val_tokens in non-overlapping batches.
    Returns mean cross-entropy loss (perplexity = exp(loss)).

    CHANGE: replaced random batch sampling with sequential iteration so
    validation loss is deterministic and fairly comparable across runs.
    """
    model.eval()
    device_type = device.type
    total_loss = 0.0
    n_batches = 0

    # Sequential pass: step through val_tokens without overlap
    B = config.batch_size
    step = B * config.context_window
    for start in range(0, len(val_tokens) - config.context_window * config.batch_size, step):
        indices = [start + b * config.context_window for b in range(config.batch_size)]
        # Guard: skip if any index goes out of bounds
        if indices[-1] + config.context_window + 1 > len(val_tokens):
            break

        x = torch.stack([val_tokens[j : j + config.context_window] for j in indices])
        y = torch.stack([val_tokens[j + 1 : j + 1 + config.context_window] for j in indices])

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")