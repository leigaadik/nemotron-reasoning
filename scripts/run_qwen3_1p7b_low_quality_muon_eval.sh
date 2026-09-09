#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/xyk/anaconda3/etc/profile.d/conda.sh
conda activate nemotron
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ACCELERATE_BYPASS_DEVICE_MAP=true PYTHONUNBUFFERED=1
mkdir -p outputs/logs
export CUDA_VISIBLE_DEVICES=1
python -u scripts/train.py --config configs/training/qwen3-1.7b-low-quality.yaml 2>&1 | tee outputs/logs/train_qwen3-1.7b-low-quality-cot-muon.log
python -u scripts/infer.py --run-name qwen3-1.7b-low-quality-cot-muon --base-model ./models/Qwen3-1.7B --adapter outputs/adapters/qwen3-1.7b-low-quality-cot-muon --suite configs/eval/current_950.yaml --out results/adapter_eval/qwen3-1.7b/qwen3-1.7b-low-quality-cot-muon/current_950 --force 2>&1 | tee outputs/logs/infer_qwen3-1.7b-low-quality-cot-muon.log
python -u scripts/score.py --suite configs/eval/current_950.yaml --raw-outputs results/adapter_eval/qwen3-1.7b/qwen3-1.7b-low-quality-cot-muon/current_950/qwen3-1.7b-low-quality-cot-muon__current_950_raw_outputs.jsonl --out results/adapter_eval/qwen3-1.7b/qwen3-1.7b-low-quality-cot-muon/current_950 2>&1 | tee outputs/logs/score_qwen3-1.7b-low-quality-cot-muon.log
