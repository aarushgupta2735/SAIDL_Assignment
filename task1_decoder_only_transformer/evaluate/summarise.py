"""
summarise.py
------------
Run this AFTER all your wandb runs are complete to auto-generate
the comparative LaTeX tables for your report.

Usage:
    python summarise.py

Outputs:
    results/summary_table_attention_pe.tex  — attention/PE sweep table
    results/summary_table_conv.tex          — conv ablation table
    results/summary_table_attention_pe.csv  — CSV for inspection
    results/summary_table_conv.csv          — CSV for inspection
"""

import os
import pandas as pd
import wandb

# ---------------------------------------------------------------
# CONFIG — adjust these to match your wandb project/entity
# ---------------------------------------------------------------
WANDB_PROJECT = "saidl-core-ml"
WANDB_ENTITY  = None   # set to your wandb username if needed

METRICS = {
    "val/perplexity":                    "Val PPL",
    "test/perplexity":                   "Test PPL",
    "perf/peak_gpu_mb":                  "Peak GPU (MB)",
    "perf/total_train_time_min":         "Train Time (min)",
    "inference/tokens_per_sec":          "Throughput (tok/s)",
    "inference/latency_ms_per_sequence": "Latency (ms/seq)",
}

CONFIG_FIELDS = ["attention", "positional_encoding", "context_window",
                 "use_conv", "conv_type", "n_decoder_layers",
                 "n_heads", "embedding_size"]
# ---------------------------------------------------------------


def fetch_runs() -> pd.DataFrame:
    api = wandb.Api()
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    runs = api.runs(path)

    rows = []
    for run in runs:
        if run.state not in ("finished", "crashed"):
            continue

        row = {"run_name": run.name}

        for field in CONFIG_FIELDS:
            row[field] = run.config.get(field, "—")

        history = run.history(keys=list(METRICS.keys()), pandas=True)
        for wandb_key, col_name in METRICS.items():
            if wandb_key in history.columns:
                if "perplexity" in wandb_key or "loss" in wandb_key:
                    row[col_name] = history[wandb_key].min()
                else:
                    row[col_name] = history[wandb_key].dropna().iloc[-1] if not history[wandb_key].dropna().empty else "—"
            else:
                row[col_name] = run.summary.get(wandb_key, "—")

        rows.append(row)

    return pd.DataFrame(rows)


def _round_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Val PPL", "Test PPL", "Peak GPU (MB)", "Train Time (min)",
                "Throughput (tok/s)", "Latency (ms/seq)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def _wrap_latex(latex: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex +
        "}%\n"
        "\\end{table}\n"
    )


def to_latex_attention_pe(df: pd.DataFrame) -> str:
    """Table for attention/PE sweep runs (use_conv=False)."""
    df = df.sort_values(["attention", "positional_encoding",
                         "context_window"]).reset_index(drop=True)

    display_cols = [
        "attention", "positional_encoding", "context_window",
        "Val PPL", "Test PPL", "Peak GPU (MB)", "Train Time (min)",
        "Throughput (tok/s)", "Latency (ms/seq)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df = _round_numerics(df[display_cols].copy())
    df = df.rename(columns={
        "attention":           "Attention",
        "positional_encoding": "PE",
        "context_window":      "Context Len",
    })

    latex = df.to_latex(
        index=False, escape=True, na_rep="—",
        column_format="l" * len(df.columns),
    )
    return _wrap_latex(
        latex,
        caption="Attention variant and positional encoding sweep on WikiText-2.",
        label="tab:attention_pe_results",
    )


def to_latex_conv(df: pd.DataFrame) -> str:
    """Table for conv ablation runs (use_conv=True)."""
    df = df.sort_values(["conv_type"]).reset_index(drop=True)

    display_cols = [
        "conv_type", "attention", "positional_encoding", "context_window",
        "Val PPL", "Test PPL", "Peak GPU (MB)", "Train Time (min)",
        "Throughput (tok/s)", "Latency (ms/seq)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df = _round_numerics(df[display_cols].copy())
    df = df.rename(columns={
        "conv_type":           "Conv Type",
        "attention":           "Attention",
        "positional_encoding": "PE",
        "context_window":      "Context Len",
    })

    latex = df.to_latex(
        index=False, escape=True, na_rep="—",
        column_format="l" * len(df.columns),
    )
    return _wrap_latex(
        latex,
        caption="Conv1D hybrid ablation (best attention + best PE) on WikiText-2.",
        label="tab:conv_results",
    )


def save(tex: str, csv_df: pd.DataFrame, tex_path: str, csv_path: str):
    with open(tex_path, "w") as f:
        f.write(tex)
    csv_df.to_csv(csv_path, index=False)
    print(f"LaTeX → {tex_path}")
    print(f"CSV   → {csv_path}")


def main():
    os.makedirs("results", exist_ok=True)

    print("Fetching runs from wandb...")
    df = fetch_runs()

    if df.empty:
        print("No finished runs found. Make sure WANDB_PROJECT is correct.")
        return

    print(f"Found {len(df)} finished runs.\n")

    # --- Split into two DataFrames ---
    # CHANGE: split on use_conv field instead of one combined table
    df_attn_pe = df[df["use_conv"] == False].copy()
    df_conv    = df[df["use_conv"] == True].copy()

    print(f"Attention/PE runs : {len(df_attn_pe)}")
    print(f"Conv runs         : {len(df_conv)}\n")

    # --- Attention / PE table ---
    if not df_attn_pe.empty:
        print(df_attn_pe[["run_name", "attention", "positional_encoding",
                           "context_window", "Val PPL"]].to_string())
        save(
            tex    = to_latex_attention_pe(df_attn_pe),
            csv_df = df_attn_pe,
            tex_path = "results/summary_table_attention_pe.tex",
            csv_path = "results/summary_table_attention_pe.csv",
        )
    else:
        print("No attention/PE runs found.")

    print()

    # --- Conv table ---
    if not df_conv.empty:
        print(df_conv[["run_name", "conv_type", "attention",
                        "positional_encoding", "Val PPL"]].to_string())
        save(
            tex    = to_latex_conv(df_conv),
            csv_df = df_conv,
            tex_path = "results/summary_table_conv.tex",
            csv_path = "results/summary_table_conv.csv",
        )
    else:
        print("No conv runs found yet — run conv experiments first.")


if __name__ == "__main__":
    main()