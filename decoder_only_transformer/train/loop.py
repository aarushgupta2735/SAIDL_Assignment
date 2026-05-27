import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig


def benchmark(model, tokens, config):
    ix = torch.randint(len(tokens) - config.context_window, (config.batch_size,), device=tokens.device)
    x = torch.stack([tokens[j:j+config.context_window] for j in ix])
    y = torch.stack([tokens[j+1:j+1+config.context_window] for j in ix])


    # warmup
    for _ in range(3):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()


    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
    torch.cuda.synchronize()


    time_per_iter = (time.time() - start) / 100
    print(f"Time per iteration: {time_per_iter:.4f}s")
    print(f"Estimated 10k iterations: {time_per_iter * 10000 / 60:.1f} minutes")




def loop(model, optimiser, tokens, config, train_config):
    for i in range(1, train_config.iterations + 1):
        ix = torch.randint(len(tokens) - config.context_window, (config.batch_size,), device=tokens.device)
        x = torch.stack([tokens[j:j+config.context_window] for j in ix])
        y = torch.stack([tokens[j+1:j+1+config.context_window] for j in ix])


        optimiser.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        update_lr(i, optimiser, config, train_config)
        optimiser.step()


        if i % 500 == 0:
            print(f"Iteration {i} : loss = {loss.item():.4f}")
            print(f"lr = {optimiser.param_groups[0]['lr']:.6f}")
            torch.save(model.state_dict(), f"experiments/checkpoints/model_checkpoint.pt")




def update_lr(i, optimiser, config, train_config):
    lr = (config.embedding_size ** -0.5) * min(i ** -0.5, i * train_config.warmup_steps ** -1.5)
    optimiser.param_groups[0]['lr'] = lr



