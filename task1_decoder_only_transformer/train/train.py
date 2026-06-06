import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import tiktoken
from task1_decoder_only_transformer.src.transformer import Transformer
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task1_decoder_only_transformer.config.train_config import TrainConfig
from task1_decoder_only_transformer.train.loop import loop
from task1_decoder_only_transformer.evaluate.logger import ExperimentLogger

def main():
    
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # --- Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Data ---
    # IMPORTANT: test set is never loaded here — only in test/test.py
    # This ensures test data is never seen during training or model selection
    print("Loading data...")
    data_dir = os.path.join(os.path.dirname(__file__), "../data")

    with open(os.path.join(data_dir, "wiki.train.txt"), encoding="utf-8") as f:
        train_data = f.read()
    with open(os.path.join(data_dir, "wiki.valid.txt"), encoding="utf-8") as f:
        val_data = f.read()

    print("Tokenizing...")
    train_tokens = torch.tensor(tokenizer.encode(train_data), dtype=torch.long, device=device)
    val_tokens   = torch.tensor(tokenizer.encode(val_data),   dtype=torch.long, device=device)
    print(f"Train tokens: {len(train_tokens):,}  |  Val tokens: {len(val_tokens):,}")

    # --- Config ---
    # NOTE: change experiment_name, attention, positional_encoding per run
    config = TransformerConfig(
        vocab_size          = tokenizer.n_vocab,
        attention           = "local", #local,sparse,mqa,standard
        positional_encoding = "sinusoidal", #rotatory,relative,attention,sinusoidal
        context_window = 1024,
        window_size=128
    )
    train_config = TrainConfig()

    # --- Model ---
    model = Transformer(config).to(device)
    # torch.compile removed — caused state_dict key prefix issues + requires Triton

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Optimiser ---
    optimiser = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        # lr=1.0 because update_lr overrides it every step via scheduler
    )

    # --- Logger ---
    logger = ExperimentLogger(config, train_config, device)

    # --- Train ---
    print(f"Starting run: {config.run_id()}")
    loop(model, optimiser, train_tokens, val_tokens, config, train_config, logger, device)

    # --- Post-training: inference latency on best checkpoint ---
    print("\nLoading best checkpoint for inference latency...")
    best_ckpt = torch.load(
        os.path.join("experiments", "checkpoints", config.run_id(), "best.pt"),
        map_location=device,
    )
    model.load_state_dict(best_ckpt["model"])
    print(f"Best val loss: {best_ckpt['val_loss']:.4f}")

    logger.log_inference_latency(model,config)

    '''# --- Post-training: sample generation ---
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
        pass'''

    logger.finish()
    print("Done. Run test/test.py to evaluate on the test set.")

if __name__ == "__main__":
    main()