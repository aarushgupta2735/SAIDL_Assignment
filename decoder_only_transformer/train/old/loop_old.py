import torch
import time
from decoder_only_transformer.evaluate.old.logger_old import ExperimentLogger
from decoder_only_transformer.evaluate.old.validate_old import evaluate


def loop(model, optimiser, train_tokens, val_tokens, config, train_config, logger: ExperimentLogger):
    best_val_loss = float("inf")

    for i in range(1, train_config.iterations + 1):

        # --- Batch sampling ---
        ix = torch.randint(
            len(train_tokens) - config.context_window,
            (config.batch_size,),
            device=train_tokens.device,
        )
        x = torch.stack([train_tokens[j : j + config.context_window] for j in ix])
        y = torch.stack([train_tokens[j + 1 : j + 1 + config.context_window] for j in ix])

        # --- Forward + backward ---
        logger.on_iter_start()

        optimiser.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        update_lr(i, optimiser, config, train_config)
        optimiser.step()

        logger.on_iter_end(
            i=i,
            train_loss=loss.item(),
            lr=optimiser.param_groups[0]["lr"],
            grad_norm=grad_norm,
        )

        # --- Validation + checkpointing every 500 steps ---
        if i % train_config.val_interval == 0:
            val_loss = evaluate(model, val_tokens, config)
            logger.log_validation(i, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.save_best(model, val_loss)

            logger.save_checkpoint(model, i, val_loss)


def update_lr(i, optimiser, config, train_config):
    lr = (config.embedding_size ** -0.5) * min(
        i ** -0.5, i * train_config.warmup_steps ** -1.5
    )
    optimiser.param_groups[0]["lr"] = lr