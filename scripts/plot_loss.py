#!/usr/bin/env python3
"""Parse Transformers Trainer logs and plot training loss curves.

Generates three PNG figures (one per dataset), each containing EMA-smoothed
loss curves for all seven Qwen3 model sizes.  Y-axis uses log scale so the
separation between model sizes stays visible throughout training
(scaling-law view).

Saves to docs/figures/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "outputs" / "logs"
FIG_DIR = REPO_ROOT / "docs" / "figures"

# ---------- log file mapping (model_key -> {dataset -> log_filename}) ----------
# Naming conventions differ across training runs, so we enumerate explicitly.

LOG_FILES: dict[str, dict[str, str]] = {
    "Qwen3-0.6B": {
        "legacy":      "train_qwen3-0.6b-legacy-cot.log",
        "low_quality": "train_qwen3-0.6b-low-quality-cot.log",
        "synthetic":   "train_qwen3-0.6b-synthetic-cot.log",
    },
    "Qwen3-1.7B": {
        "legacy":      "train_qwen3-1.7b-legacy-cot.log",
        "low_quality": "train_qwen3-1.7b-low-quality-cot.log",
        "synthetic":   "train_qwen3-1.7b-synthetic-cot.log",
    },
    "Qwen3-4B": {
        "legacy":      "train_qwen3-4b-legacy-cot.log",
        "low_quality": "train_qwen3-4b-low-quality-cot.log",
        "synthetic":   "train_qwen3-4b-synthetic-cot.log",
    },
    "Qwen3-8B": {
        "legacy":      "train_qwen3-8b-legacy-cot.log",
        "low_quality": "train_qwen3-8b-low-quality-cot.log",
        "synthetic":   "train_qwen3-8b-synthetic-cot.log",
    },
    "Qwen3-14B": {
        "legacy":      "train_qwen3-14b-legacy.log",
        "low_quality": "train_qwen3-14b-low-quality.log",
        "synthetic":   "train_qwen3-14b-synthetic.log",
    },
    "Qwen3-30B-A3B": {
        "legacy":      "qwen3-30b-a3b-legacy-cot.log",
        "low_quality": "qwen3-30b-a3b-low-quality-cot.log",
        "synthetic":   "qwen3-30b-a3b-synthetic-cot.log",
    },
    "Qwen3-32B": {
        "legacy":      "qwen3-32b-legacy-cot.log",
        "low_quality": "qwen3-32b-low-quality-cot.log",
        "synthetic":   "qwen3-32b-synthetic-cot.log",
    },
}

DATASET_LABELS = {
    "legacy":      "Legacy CoT",
    "low_quality": "Low-Quality CoT",
    "synthetic":   "Synthetic CoT",
}

DATASET_FIG_NAMES = {
    "legacy":      "training_loss_legacy",
    "low_quality": "training_loss_low_quality",
    "synthetic":   "training_loss_synthetic",
}

# Ordered from small to large for legend readability.
MODEL_ORDER = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-30B-A3B",
    "Qwen3-32B",
]

# Distinct colors for seven models.
MODEL_COLORS = {
    "Qwen3-0.6B":    "#e6194b",
    "Qwen3-1.7B":    "#f58231",
    "Qwen3-4B":      "#ffe119",
    "Qwen3-8B":      "#3cb44b",
    "Qwen3-14B":     "#4363d8",
    "Qwen3-30B-A3B": "#911eb4",
    "Qwen3-32B":     "#000075",
}

# Regex to capture the dict-like log line produced by Transformers Trainer.
_LOG_RE = re.compile(r"\{[^{}]*'loss':\s*[\d.e+-]+[^{}]*\}")

# EMA smoothing factor — higher means smoother (0 = no smoothing, 1 = flat).
EMA_ALPHA = 0.9


def parse_loss(log_path: Path) -> list[tuple[int, float]]:
    """Return [(step, loss), ...] extracted from a Trainer log file."""
    text = log_path.read_text(errors="replace")
    entries: list[tuple[int, float]] = []
    for match in _LOG_RE.finditer(text):
        try:
            record = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            continue
        if "loss" in record:
            # Trainer logs every 10 steps (the default logging_steps).
            step = (len(entries) + 1) * 10
            entries.append((step, float(record["loss"])))
    return entries


def ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


def plot_dataset(dataset_key: str, smooth: bool = False) -> None:
    """Generate one figure for a given dataset, covering all model sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODEL_ORDER:
        log_name = LOG_FILES[model][dataset_key]
        log_path = LOG_DIR / log_name
        if not log_path.exists():
            print(f"  SKIP {model}: {log_path} not found")
            continue
        entries = parse_loss(log_path)
        if not entries:
            print(f"  SKIP {model}: no loss entries in {log_path.name}")
            continue
        steps = [s for s, _ in entries]
        losses = [l for _, l in entries]
        if smooth:
            losses = ema_smooth(losses, EMA_ALPHA)
        ax.plot(steps, losses, label=model, color=MODEL_COLORS[model], linewidth=1.4)

    suffix = " (EMA Smoothed)" if smooth else ""
    ax.set_yscale("log")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title(f"Training Loss — {DATASET_LABELS[dataset_key]}{suffix}", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    stem = DATASET_FIG_NAMES[dataset_key]
    fname = f"{stem}_smooth.png" if smooth else f"{stem}.png"
    out_path = FIG_DIR / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_combined() -> None:
    """2×3 grid: rows = raw | smoothed, cols = dataset quality worst→best."""
    datasets = ["low_quality", "legacy", "synthetic"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    fig.suptitle("Training Loss — All Datasets & Model Sizes", fontsize=16, y=1.01)

    col_titles = [DATASET_LABELS[ds] for ds in datasets]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, pad=8)

    row_labels = ["Raw Loss", "EMA-Smoothed Loss"]
    for row, smooth in enumerate([False, True]):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            for model in MODEL_ORDER:
                log_name = LOG_FILES[model][ds]
                log_path = LOG_DIR / log_name
                if not log_path.exists():
                    continue
                entries = parse_loss(log_path)
                if not entries:
                    continue
                steps = [s for s, _ in entries]
                losses = [l for _, l in entries]
                if smooth:
                    losses = ema_smooth(losses, EMA_ALPHA)
                ax.plot(steps, losses, label=model, color=MODEL_COLORS[model], linewidth=1.2)

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel("Training Step", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\n\nLoss (log scale)", fontsize=10)
            else:
                ax.set_ylabel("Loss (log scale)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    out_path = FIG_DIR / "training_loss_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")



def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ds in ("legacy", "low_quality", "synthetic"):
        print(f"Plotting {DATASET_LABELS[ds]} (raw) ...")
        plot_dataset(ds, smooth=False)
        print(f"Plotting {DATASET_LABELS[ds]} (smoothed) ...")
        plot_dataset(ds, smooth=True)
    print("Plotting combined figure ...")
    plot_combined()
    print("Done.")


if __name__ == "__main__":
    main()
