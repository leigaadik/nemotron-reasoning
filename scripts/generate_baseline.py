#!/usr/bin/env python3
"""Stage 1: generate raw model outputs for a single baseline model.

Loads the fixed validation split, drives vLLM through
`src.providers.local_vllm.LocalVLLMProvider`, and streams each raw
generation to `<model>_raw_outputs.jsonl` in the output directory.
Answer extraction and scoring live in `scripts/evaluate_baseline.py` so
the expensive inference does not have to be redone when the scoring
rules change.

All inference parameters (sampling, vLLM engine flags, chat template
kwargs, boxed-answer prompt suffix) are hard-coded below to keep every
baseline model on the same footing — the only per-model inputs are the
name (used purely as an artifact prefix) and the local weights path.
The full parameter dict is snapshotted into `<model>_run.yaml` next to
the raw outputs so any future comparison can inspect what was used.

Usage:
    python scripts/generate_baseline.py \
        --model-name qwen3-30b-a3b \
        --model-path <REPO_ROOT>/models/Qwen3-30B-A3B \
        --out results/prompting_baselines/qwen3-30b-a3b \
        [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.dataset import load_validation  # noqa: E402
from src.providers import build_provider  # noqa: E402


# ---------------------------------------------------------------------------
# Baseline inference parameters. Shared across all models so accuracy
# differences are attributable to the model itself, not the runtime
# configuration. Edit here (not per-model) if you want to sweep them.
# ---------------------------------------------------------------------------

PROMPT_SUFFIX = (
    'Please put your final answer inside `\\boxed{}`. '
    'For example: `\\boxed{your answer}`'
)

CHAT_KWARGS = {
    # apply_chat_template kwarg; provider falls back to unset if the
    # tokenizer does not accept it.
    'enable_thinking': True,
    # No system prompt — matches adapter_validation.ipynb.
    'system': None,
}

SAMPLING_PARAMS = {
    'temperature': 0.0,
    'top_p': 1.0,
    'max_tokens': 32768,
}

VLLM_ENGINE_PARAMS = {
    'tensor_parallel_size': 1,
    'max_num_seqs': 32,
    'gpu_memory_utilization': 0.85,
    'max_model_len': 32768,
    'dtype': 'auto',
    'trust_remote_code': True,
    'enable_prefix_caching': True,
    'enable_chunked_prefill': True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-name',
        required=True,
        help='Short model identifier, used as the prefix for output files.',
    )
    parser.add_argument(
        '--model-path',
        required=True,
        help='Absolute path to the local model weights directory.',
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output directory (created if missing).',
    )
    parser.add_argument(
        '--train',
        default='data/train.csv',
        help='Path to train.csv (default: data/train.csv).',
    )
    parser.add_argument(
        '--validation-ids',
        default='configs/eval/validation_ids_seed42_size950.csv',
        help='Path to the fixed validation id list.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite an existing <model>_raw_outputs.jsonl in --out.',
    )
    return parser.parse_args()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def resolve_input_path(user_path: str) -> Path:
    if os.path.isabs(user_path):
        return Path(user_path)
    return (REPO_ROOT / user_path).resolve()


def main() -> None:
    args = parse_args()

    model_name = args.model_name
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'model path does not exist: {model_path}')

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f'{model_name}_raw_outputs.jsonl'
    run_yaml_path = out_dir / f'{model_name}_run.yaml'

    if raw_path.exists() and not args.force:
        raise FileExistsError(
            f'{raw_path} already exists. Pass --force to overwrite.'
        )

    # Early write-probe: fail fast if the volume is full instead of after
    # a multi-hour inference run.
    probe = out_dir / '.write_probe'
    probe.write_text('ok', encoding='utf-8')
    probe.unlink()

    train_csv = resolve_input_path(args.train)
    ids_csv = resolve_input_path(args.validation_ids)

    examples = load_validation(train_csv, ids_csv)

    # Build the parameter bundle the provider consumes. This is the same
    # dict shape we used to load from YAML — keeping it lets us switch
    # back to per-model configs later without touching the provider.
    provider_cfg = {
        'name': model_name,
        'model_path': str(model_path),
        'provider': 'local_vllm',
        'chat': CHAT_KWARGS,
        'prompt_suffix': PROMPT_SUFFIX,
        'sampling': SAMPLING_PARAMS,
        'vllm': VLLM_ENGINE_PARAMS,
    }

    run_meta = {
        'model': model_name,
        'model_path': str(model_path),
        'train_csv': str(train_csv),
        'validation_ids': str(ids_csv),
        'num_examples': len(examples),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(REPO_ROOT),
        'params': {
            'prompt_suffix': PROMPT_SUFFIX,
            'chat': CHAT_KWARGS,
            'sampling': SAMPLING_PARAMS,
            'vllm': VLLM_ENGINE_PARAMS,
        },
    }
    with run_yaml_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(run_meta, f, sort_keys=False, allow_unicode=True)

    provider = build_provider(provider_cfg)
    try:
        print(
            f'[generate] model={model_name} n={len(examples)} '
            f'out={raw_path}',
            flush=True,
        )
        results = provider.generate(examples, PROMPT_SUFFIX)
    finally:
        provider.close()

    with raw_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        'id': r.id,
                        'prompt': r.prompt,
                        'raw_output': r.raw_output,
                        'finish_reason': r.finish_reason,
                        'tokens_out': r.tokens_out,
                    },
                    ensure_ascii=False,
                )
            )
            f.write('\n')

    print(f'[generate] wrote {len(results)} records to {raw_path}')


if __name__ == '__main__':
    main()
