# train/loop.py
import torch
import time

def get_batch(tokens, config, device):
    B, T = config.batch_size, config.context_window
    ix = torch.randint(0, len(tokens) - T - 1, (B,))
    xt = torch.stack([tokens[i   : i+T  ] for i in ix]).to(device)
    yt = torch.stack([tokens[i+1 : i+T+1] for i in ix]).to(device)
    return xt, yt

def update_lr(optimiser, step, config):
    lr = (config.embedding_size ** -0.5) * min(
        step ** -0.5,
        step * config.warmup_steps ** -1.5
    )
    for group in optimiser.param_groups:
        group['lr'] = lr

def loop(model, optimiser, tokens, config, device):

    # ------------------------------------------------------------------ #
    #  one warmup forward+backward before timing starts                   #
    #  GPUs are slow on the first call due to kernel compilation          #
    # ------------------------------------------------------------------ #
    model.train()
    xt, yt = get_batch(tokens, config, device)
    _, loss = model(xt, yt)
    loss.backward()
    optimiser.zero_grad()
    print("warmup done")

    # ------------------------------------------------------------------ #
    #  timing events                                                       #
    # ------------------------------------------------------------------ #
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end   = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end   = torch.cuda.Event(enable_timing=True)
    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end   = torch.cuda.Event(enable_timing=True)

    wall_start = time.time()

    for step in range(1, config.iterations + 1):

        # ---------- data ----------
        t0 = time.time()
        xt, yt = get_batch(tokens, config, device)
        t1 = time.time()
        data_ms = (t1 - t0) * 1000

        # ---------- forward ----------
        optimiser.zero_grad()
        fwd_start.record()
        logits, loss = model(xt, yt)
        fwd_end.record()

        # ---------- backward ----------
        bwd_start.record()
        loss.backward()
        bwd_end.record()

        # ---------- clip + update ----------
        opt_start.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        update_lr(optimiser, step, config)
        optimiser.step()
        opt_end.record()

        # ---------- sync + print ----------
        if step % 100 == 0:
            torch.cuda.synchronize()

            fwd_ms = fwd_start.elapsed_time(fwd_end)
            bwd_ms = bwd_start.elapsed_time(bwd_end)
            opt_ms = opt_start.elapsed_time(opt_end)
            total_ms = fwd_ms + bwd_ms + opt_ms + data_ms

            tokens_per_sec = (config.batch_size * config.context_window) / (total_ms / 1000)

            print(
                f"step {step:>6} | "
                f"loss {loss.item():>7.4f} | "
                f"lr {optimiser.param_groups[0]['lr']:.6f} | "
                f"data {data_ms:>6.1f}ms | "
                f"fwd {fwd_ms:>6.1f}ms | "
                f"bwd {bwd_ms:>6.1f}ms | "
                f"opt {opt_ms:>6.1f}ms | "
                f"total {total_ms:>6.1f}ms | "
                f"tok/s {tokens_per_sec:>8.0f}"
            )

        # ---------- eval ----------
        if step % 1000 == 0:
            wall_elapsed = time.time() - wall_start
            print(f"\n--- step {step} | wall time {wall_elapsed:.1f}s ---\n")

    print("training complete")