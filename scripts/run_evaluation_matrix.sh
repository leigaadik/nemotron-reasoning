#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 BASE_MODEL LEGACY_ADAPTER SYNTHETIC_ADAPTER LOW_QUALITY_ADAPTER OUTPUT_ROOT" >&2
  exit 2
fi

BASE_MODEL=$1
LEGACY_ADAPTER=$2
SYNTHETIC_ADAPTER=$3
LOW_QUALITY_ADAPTER=$4
OUTPUT_ROOT=$5

mkdir -p "$OUTPUT_ROOT"
export NEMOTRON_EVAL_DISABLE_BROKEN_EXTENSIONS=1
export PYTHONPATH="$PWD/scripts/eval_compat${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$OUTPUT_ROOT/evaluation_matrix.log") 2>&1

run_one() {
  local run_name=$1
  local adapter=$2
  local suite=$3
  local suite_config="configs/eval/${suite}.yaml"
  local out="$OUTPUT_ROOT/$run_name/$suite"
  local raw="$out/${run_name}__${suite}_raw_outputs.jsonl"
  mkdir -p "$out"
  echo "[$(date -Iseconds)] START adapter=$run_name suite=$suite"
  python scripts/infer.py \
    --run-name "$run_name" \
    --base-model "$BASE_MODEL" \
    --adapter "$adapter" \
    --suite "$suite_config" \
    --out "$out" \
    --force 2>&1 | tee "$out/inference.log"
  python scripts/score.py \
    --suite "$suite_config" \
    --raw-outputs "$raw" \
    --out "$out" 2>&1 | tee "$out/scoring.log"
  echo "[$(date -Iseconds)] DONE adapter=$run_name suite=$suite"
}

run_one legacy-cot-transformers "$LEGACY_ADAPTER" current_950
run_one synthetic-cot-transformers "$SYNTHETIC_ADAPTER" current_950
run_one low-quality-cot "$LOW_QUALITY_ADAPTER" current_950

python scripts/summarize_matrix.py --root "$OUTPUT_ROOT" \
  2>&1 | tee "$OUTPUT_ROOT/matrix_summary.log"
