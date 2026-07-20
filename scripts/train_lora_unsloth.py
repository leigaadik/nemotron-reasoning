#!/usr/bin/env python3
"""LoRA training entrypoint.

This script currently provides the configuration boundary for the LoRA
training pipeline. It reads one YAML file, prints the exact resolved path
and parameter snapshot, then either exits in dry-run mode or continues to
the training implementation as it is filled in from the reference notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.config import format_config, load_training_config  # noqa: E402


DEFAULT_CONFIG = "configs/training/lora_unsloth_nemotron_30b_a3b.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Training YAML config path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and print the resolved training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_training_config(args.config)

    print(f"[train_lora_unsloth] config_path: {config_path}", flush=True)
    print("[train_lora_unsloth] config:", flush=True)
    print(format_config(config), flush=True)

    if args.dry_run:
        print("[train_lora_unsloth] dry_run=true, skip training.", flush=True)
        return

    raise NotImplementedError(
        "Training implementation is not wired yet. "
        "Use --dry-run to validate the YAML config boundary first."
    )


if __name__ == "__main__":
    main()
