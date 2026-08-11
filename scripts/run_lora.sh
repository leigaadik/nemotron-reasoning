#!/usr/bin/env bash
# Template script for a single LoRA training run.
#
# Usage:
#   1. Copy configs/training/legacy.yaml to configs/training/<experiment>.yaml
#   2. Edit the new config: update experiment.name, paths.train_jsonl,
#      paths.adapter_dir, and training.output_dir.
#   3. Set CONFIG and LOG below to match, then run this script inside a GPU
#      container.
#
# Pre-requisite: training data already built under outputs/data/.
#   If not, run:  python scripts/build_data.py --config configs/data/<dataset>.yaml

set -euo pipefail

# --- Edit these two lines for each experiment ---
CONFIG=configs/training/legacy.yaml
LOG=outputs/logs/train_legacy.log
# -------------------------------------------------

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCELERATE_BYPASS_DEVICE_MAP=true
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs

python -u scripts/train.py --config "$CONFIG" 2>&1 | tee "$LOG"
