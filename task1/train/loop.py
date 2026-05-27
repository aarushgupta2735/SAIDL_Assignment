# train/loop.py
import torch
import time

def get_batch(tokens, config, device):
    T  = config.context_window
    B  = config.batch_size
    ix = torch.randint(0, len(tokens) - T - 1, (B,))
    xt = torch.stack([tokens[j   : j+T  ] for j in ix]).to(device)
    yt = torch.stack([tokens[j+1 : j+T+1] for j in ix]).to(device)
    return xt, yt

def update_lr(step, optimiser, config, train_config):
    lr = (config.embedding_size ** -0.5) * min(
        step ** -0.5,                            # ← decay phase
        step * train_config.warmup_steps ** -1.5       # ← warmup phase
    )
    for group in optimiser.param_groups:
        group['lr'] = lr

def loop(model, optimiser, tokens, device,config,train_config):
    model.train()

    # warmup forward pass — GPU kernel compilation on first call
    # means first real step is always slow without this
    xt, yt = get_batch(tokens, config, device)
    _, loss = model(xt, yt)
    loss.backward()
    optimiser.zero_grad()
    print("warmup done")

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end   = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end   = torch.cuda.Event(enable_timing=True)

    for step in range(1, train_config.iterations + 1):

        t0 = time.time()
        xt, yt = get_batch(tokens, config, device)
        data_ms = (time.time() - t0) * 1000

        optimiser.zero_grad()

        fwd_start.record()
        logits, loss = model(xt, yt)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        update_lr(step, optimiser, config, train_config)
        optimiser.step()

        if step % 100 == 0:
            torch.cuda.synchronize()
            fwd_ms   = fwd_start.elapsed_time(fwd_end)
            bwd_ms   = bwd_start.elapsed_time(bwd_end)
            total_ms = data_ms + fwd_ms + bwd_ms
            tok_s    = (config.batch_size * config.context_window) / (total_ms / 1000)

            print(
                f"step {step:>6} | "
                f"loss {loss.item():>7.4f} | "
                f"lr {optimiser.param_groups[0]['lr']:.2e} | "
                f"data {data_ms:>5.1f}ms | "
                f"fwd {fwd_ms:>6.1f}ms | "
                f"bwd {bwd_ms:>6.1f}ms | "
                f"total {total_ms:>6.1f}ms | "
                f"tok/s {tok_s:>8.0f}")