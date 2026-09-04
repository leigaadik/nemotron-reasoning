#!/usr/bin/env python3
"""Plot training loss for the module ablation experiments (m4/m5/m7/m8).

Layout: 1 row × 3 columns (one per dataset).
Each subplot shows 4 EMA-smoothed curves, one per module config.
Saves docs/figures/module_ablation_loss.png.
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

LOG_FILES = {
    "m4": {
        "legacy":      "module-train-m4-r16-legacy.log",
        "low_quality": "module-train-m4-r16-lq.log",
        "synthetic":   "module-train-m4-r16-synthetic.log",
    },
    "m5": {
        "legacy":      "module-train-m5-r16-legacy.log",
        "low_quality": "module-train-m5-r16-lq.log",
        "synthetic":   "module-train-m5-r16-synthetic.log",
    },
    "m7": {
        "legacy":      "module-train-m7-r16-legacy.log",
        "low_quality": "module-train-m7-r16-lq.log",
        "synthetic":   "module-train-m7-r16-synthetic.log",
    },
    "m8": {
        "legacy":      "module-train-m8-r16-legacy.log",
        "low_quality": "module-train-m8-r16-lq.log",
        "synthetic":   "module-train-m8-r16-synthetic.log",
    },
}

MODULE_LABELS = {
    "m4": "m4: attn only\n(q/k/v/o_proj)",
    "m5": "m5: attn + router\n(+mlp.gate)",
    "m7": "m7: attn + FFN\n(+gate/up/down_proj)",
    "m8": "m8: attn + FFN + router",
}

MODULE_COLORS = {
    "m4": "#e6194b",
    "m5": "#f58231",
    "m7": "#3cb44b",
    "m8": "#4363d8",
}

DATASET_LABELS = {
    "legacy":      "Legacy CoT",
    "low_quality": "Low-Quality CoT",
    "synthetic":   "Synthetic CoT",
}

_LOG_RE = re.compile(r"\{[^{}]*'loss':\s*[\d.e+-]+[^{}]*\}")
EMA_ALPHA = 0.9


def parse_loss(log_path: Path) -> list[tuple[int, float]]:
    text = log_path.read_text(errors="replace")
    entries: list[tuple[int, float]] = []
    for match in _LOG_RE.finditer(text):
        try:
            record = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            continue
        if "loss" in record:
            step = (len(entries) + 1) * 10
            entries.append((step, float(record["loss"])))
    return entries


def ema_smooth(values: list[float], alpha: float) -> list[float]:
    smoothed, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["low_quality", "legacy", "synthetic"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Training Loss — Module Ablation (Qwen3-30B-A3B, r=16)",
        fontsize=14, y=1.01,
    )

    col_titles = [DATASET_LABELS[ds] for ds in datasets]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, pad=8)

    row_labels = ["Raw Loss", "EMA-Smoothed Loss"]
    for row, smooth in enumerate([False, True]):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            for mod in ("m4", "m5", "m7", "m8"):
                log_path = LOG_DIR / LOG_FILES[mod][ds]
                if not log_path.exists():
                    print(f"  SKIP {mod}/{ds}: not found")
                    continue
                entries = parse_loss(log_path)
                if not entries:
                    continue
                steps = [s for s, _ in entries]
                losses = [l for _, l in entries]
                if smooth:
                    losses = ema_smooth(losses, EMA_ALPHA)
                ax.plot(steps, losses, label=MODULE_LABELS[mod],
                        color=MODULE_COLORS[mod], linewidth=1.6)

            ax.set_yscale("log")
            ax.set_xlabel("Training Step", fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\n\nLoss (log scale)", fontsize=10)
            else:
                ax.set_ylabel("Loss (log scale)", fontsize=10)
            ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out_path = FIG_DIR / "module_ablation_loss.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
