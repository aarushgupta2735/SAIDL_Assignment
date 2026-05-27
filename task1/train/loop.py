import torch
from config.train_config import TrainConfig
from config.transformer_config import TransformerConfig

# Using a Gradient Scaler for Automatic Mixed Precision (AMP)
scaler = torch.amp.GradScaler('cuda')

def loop(model, optimiser, tokens, n):
    model.train()
    
    # Pre-allocate the offsets for extremely fast slicing 
    T = TransformerConfig.context_window
    B = TransformerConfig.batch_size
    offsets = torch.arange(T, device=tokens.device)

    for i in range(1, n+1):
        # 1. FASTER BATCHING: Avoid torch.stack and list comprehensions
        ix = torch.randint(len(tokens) - T - 1, (B,))
        
        # Broadcast indices directly into the token array (Single CUDA operation)
        idx = ix.unsqueeze(1) + offsets
        x = tokens[idx]
        y = tokens[idx + 1]
        
        # 2. FASTER ZERO GRAD
        optimiser.zero_grad(set_to_none=True)
        
        # 3. AUTOMATIC MIXED PRECISION (AMP): Runs ops in FP16/BF16 where possible
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, y)
        
        # 4. SCALED BACKPROPAGATION
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        update_lr(i, optimiser)
        
        # Update optimizer through scaler
        scaler.step(optimiser)
        scaler.update()
        
        if(i%100==0):
            print(f"Iteration {i} : loss = {loss.item():.4f} | lr = {optimiser.param_groups[0]['lr']:.6f}")

def update_lr(i,optimiser):
    optimiser.param_groups[0]['lr'] = (TransformerConfig.embedding_size**-0.5)*min(i**0.5, i*TrainConfig.warmup_steps**-1.5)