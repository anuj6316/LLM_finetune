#!/usr/bin/env python3
"""
Token Distribution Analysis for Gemma 4 E4B IT Fine-Tuning Dataset.

Analyzes token counts across system/user/assistant roles using the actual
Gemma chat template (including special tokens like <bos>, role markers, etc.).

Outputs:
  - Console: detailed statistics table
  - 6 PNG charts saved to dataset/processed/bronze/token_analysis/
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-4-E4B-it"
DATASET_PATH = Path(__file__).parent.parent / "dataset" / "processed" / "bronze" / "processed_bronze_dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "processed" / "bronze" / "token_analysis"
MAX_SEQ_LENGTH = 1024

THRESHOLDS = [512, 768, 1024, 1536, 2048]
ROLE_COLORS = {"system": "#4C72B0", "user": "#55A868", "assistant": "#C44E52"}


# ── TOKENIZATION HELPERS ──────────────────────────────────────────────────────
def count_tokens(tokenizer, messages):
    """Count tokens for a list of messages using the full chat template."""
    return len(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )["input_ids"]
    )


def count_tokens_cumulative(tokenizer, conversation):
    """
    Tokenize each role cumulatively to get accurate per-role token counts
    that account for template overhead (special tokens, role markers, etc.).

    Returns dict with system/user/assistant/total token counts.
    """
    system_msg = [m for m in conversation if m["role"] == "system"]
    user_msg = [m for m in conversation if m["role"] == "user"]
    assistant_msg = [m for m in conversation if m["role"] == "assistant"]

    system_tokens = count_tokens(tokenizer, system_msg) if system_msg else 0
    user_tokens = count_tokens(tokenizer, system_msg + user_msg) - system_tokens
    assistant_tokens = (
        count_tokens(tokenizer, system_msg + user_msg + assistant_msg)
        - count_tokens(tokenizer, system_msg + user_msg)
    )
    total = system_tokens + user_tokens + assistant_tokens

    return {
        "system": system_tokens,
        "user": user_tokens,
        "assistant": assistant_tokens,
        "total": total,
    }


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_dataset(path):
    """Load JSONL dataset."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ── STATISTICS ────────────────────────────────────────────────────────────────
def compute_statistics(df):
    """Compute detailed statistics for each role and total."""
    stats = {}
    for col in ["system", "user", "assistant", "total"]:
        series = df[col]
        stats[col] = {
            "count": int(series.count()),
            "min": int(series.min()),
            "max": int(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "P25": float(series.quantile(0.25)),
            "P75": float(series.quantile(0.75)),
            "P90": float(series.quantile(0.90)),
            "P95": float(series.quantile(0.95)),
            "P99": float(series.quantile(0.99)),
        }
    return stats


def print_statistics(stats, df, max_seq_len):
    """Print formatted statistics to console."""
    print("\n" + "=" * 80)
    print("TOKEN DISTRIBUTION ANALYSIS — Gemma 4 E4B IT")
    print("=" * 80)

    print(f"\nDataset: {len(df)} samples")
    print(f"Model context window: {max_seq_len} tokens")

    # Per-role statistics
    header = f"{'Metric':<12} {'System':>10} {'User':>10} {'Assistant':>10} {'Total':>10}"
    print(f"\n{'─' * 55}")
    print(header)
    print(f"{'─' * 55}")

    for metric in ["min", "max", "mean", "median", "std", "P25", "P75", "P90", "P95", "P99"]:
        row = f"{metric:<12}"
        for col in ["system", "user", "assistant", "total"]:
            row += f" {stats[col][metric]:>10.1f}"
        print(row)

    print(f"{'─' * 55}")

    # Role contribution
    print(f"\n{'Role Contribution to Total Tokens':^55}")
    print(f"{'─' * 55}")
    for role in ["system", "user", "assistant"]:
        pct = stats[role]["mean"] / stats["total"]["mean"] * 100
        print(f"  {role:<12} {stats[role]['mean']:>8.1f} tokens  ({pct:>5.1f}%)")
    print(f"  {'TOTAL':<12} {stats['total']['mean']:>8.1f} tokens")
    print(f"{'─' * 55}")

    # Threshold analysis
    print(f"\n{'Threshold Analysis (MAX_SEQ_LENGTH = ' + str(max_seq_len) + ')':^55}")
    print(f"{'─' * 55}")
    for threshold in THRESHOLDS:
        within = (df["total"] <= threshold).sum()
        pct = within / len(df) * 100
        truncated = len(df) - within
        print(f"  ≤ {threshold:<5} tokens: {within:>5} samples ({pct:>5.1f}%) | {truncated} truncated")

    # Effective dataset size
    within_limit = (df["total"] <= max_seq_len).sum()
    total_tokens_raw = df["total"].sum()
    total_tokens_effective = df["total"].clip(upper=max_seq_len).sum()
    print(f"\n{'─' * 55}")
    print(f"  Samples within {max_seq_len} limit:  {within_limit}/{len(df)} ({within_limit/len(df)*100:.1f}%)")
    print(f"  Total tokens (raw):              {total_tokens_raw:,}")
    print(f"  Total tokens (after truncation): {total_tokens_effective:,}")
    print(f"  Tokens lost to truncation:       {total_tokens_raw - total_tokens_effective:,}")
    print(f"{'─' * 55}\n")


# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
def create_visualizations(df, stats, max_seq_len, output_dir):
    """Generate and save 6 analysis charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Chart 1: Total Token Histogram ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["total"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(max_seq_len, color="#C44E52", linestyle="--", linewidth=2, label=f"MAX_SEQ_LENGTH = {max_seq_len}")
    ax.axvline(stats["total"]["mean"], color="#55A868", linestyle="-.", linewidth=2, label=f"Mean = {stats['total']['mean']:.0f}")
    ax.axvline(stats["total"]["median"], color="#DD8452", linestyle=":", linewidth=2, label=f"Median = {stats['total']['median']:.0f}")
    ax.set_xlabel("Total Tokens per Sample")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Total Token Counts per Sample")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "01_total_token_histogram.png", dpi=150)
    plt.close(fig)

    # ── Chart 2: Per-Role Box Plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    role_data = df[["system", "user", "assistant"]].melt(var_name="Role", value_name="Tokens")
    palette = ROLE_COLORS
    sns.boxplot(data=role_data, x="Role", y="Tokens", hue="Role", palette=palette, ax=ax, showfliers=True, flierprops={"markersize": 3}, legend=False)
    ax.set_title("Token Count Distribution by Role")
    ax.set_ylabel("Token Count")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "02_role_token_boxplot.png", dpi=150)
    plt.close(fig)

    # ── Chart 3: Stacked Bar — Average Token Split ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    roles = ["system", "user", "assistant"]
    means = [stats[r]["mean"] for r in roles]
    colors = [ROLE_COLORS[r] for r in roles]
    bars = ax.bar(["System", "User", "Assistant"], means, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean Token Count")
    ax.set_title("Average Token Allocation by Role")
    ax.set_ylim(0, max(means) * 1.25)
    plt.tight_layout()
    fig.savefig(output_dir / "03_role_token_stacked_bar.png", dpi=150)
    plt.close(fig)

    # ── Chart 4: Cumulative Distribution (CDF) ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_totals = np.sort(df["total"].values)
    cdf = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals) * 100
    ax.plot(sorted_totals, cdf, linewidth=2, color="#4C72B0")
    ax.fill_between(sorted_totals, cdf, alpha=0.15, color="#4C72B0")
    for threshold in THRESHOLDS:
        pct = (df["total"] <= threshold).sum() / len(df) * 100
        ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(f"{threshold}t\n({pct:.0f}%)", xy=(threshold, pct), xytext=(threshold + 50, pct - 8),
                     fontsize=9, color="gray", fontweight="bold")
    ax.axvline(max_seq_len, color="#C44E52", linestyle="--", linewidth=2)
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Cumulative % of Samples")
    ax.set_title("Cumulative Distribution of Total Tokens")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, min(sorted_totals[-1], 3000))
    plt.tight_layout()
    fig.savefig(output_dir / "04_cumulative_distribution.png", dpi=150)
    plt.close(fig)

    # ── Chart 5: User vs Assistant Scatter ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["user"], df["assistant"],
        c=df["system"], cmap="viridis", alpha=0.5, s=20, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax, label="System Tokens")
    ax.axhline(max_seq_len, color="#C44E52", linestyle="--", alpha=0.7, label=f"MAX_SEQ = {max_seq_len}")
    ax.axvline(max_seq_len, color="#C44E52", linestyle="--", alpha=0.7)
    ax.set_xlabel("User Tokens")
    ax.set_ylabel("Assistant Tokens")
    ax.set_title("User vs Assistant Token Count (color = System tokens)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "05_user_vs_assistant_scatter.png", dpi=150)
    plt.close(fig)

    # ── Chart 6: Threshold Analysis Bar ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    within_counts = [(df["total"] <= t).sum() for t in THRESHOLDS]
    truncated_counts = [len(df) - c for c in within_counts]
    x = np.arange(len(THRESHOLDS))
    width = 0.5
    bars_within = ax.bar(x, within_counts, width, label="Within limit", color="#55A868", edgecolor="white")
    bars_trunc = ax.bar(x, truncated_counts, width, bottom=within_counts, label="Truncated", color="#C44E52", edgecolor="white")
    for i, (w, t) in enumerate(zip(within_counts, truncated_counts)):
        pct = w / len(df) * 100
        ax.text(i, w / 2, f"{pct:.1f}%", ha="center", va="center", fontweight="bold", fontsize=11, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤ {t}" for t in THRESHOLDS])
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Samples Within Token Thresholds (total: {len(df)})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "06_threshold_analysis.png", dpi=150)
    plt.close(fig)

    print(f"Charts saved to: {output_dir}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading dataset: {DATASET_PATH}...")
    raw_data = load_dataset(DATASET_PATH)
    print(f"Found {len(raw_data)} samples. Tokenizing...")

    records = []
    for i, sample in enumerate(raw_data):
        conversation = sample["conversations"]
        tokens = count_tokens_cumulative(tokenizer, conversation)
        records.append(tokens)

        if (i + 1) % 500 == 0 or (i + 1) == len(raw_data):
            print(f"  Processed {i + 1}/{len(raw_data)} samples...")

    df = pd.DataFrame(records)
    stats = compute_statistics(df)
    print_statistics(stats, df, MAX_SEQ_LENGTH)
    create_visualizations(df, stats, MAX_SEQ_LENGTH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
