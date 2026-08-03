#!/usr/bin/env python3
"""Stage 1: generate raw model outputs for a LoRA adapter run.

Same shape as scripts/generate_baseline.py, but drives vLLM with a
LoRARequest so `base + adapter` is what actually generates answers. The
output layout is identical (`<model>_raw_outputs.jsonl` +
`<model>_run.yaml`), which lets `scripts/evaluate_baseline.py` score
adapter runs and baseline runs with the same code path.

All inference parameters (sampling, vLLM engine flags, chat template
kwargs, boxed-answer prompt suffix) are hard-coded below and kept
identical to `scripts/generate_baseline.py`, so LoRA vs baseline diffs
are attributable to the adapter, not the runtime configuration. The
only per-run inputs are the model slug, the base model path, and the
adapter path.

Usage:
    python scripts/evaluate_adapter.py \
        --model-name qwen3-30b-a3b-lora \
        --base-model ./models/Qwen3-30B-A3B \
        --adapter outputs/lora_finetuning/qwen3_30b_a3b/adapter \
        --out results/lora_finetuning/qwen3-30b-a3b-lora \
        [--force]

Score with the shared stage-2 evaluator afterwards:
    python scripts/evaluate_baseline.py \
        --run-dir results/lora_finetuning/qwen3-30b-a3b-lora
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
# Inference parameters. Held identical to scripts/generate_baseline.py so
# any accuracy delta between LoRA and baseline runs is attributable to the
# adapter itself. Edit both files together if you want to sweep them.
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
    'max_tokens': 7680,
}

VLLM_ENGINE_PARAMS = {
    'tensor_parallel_size': 1,
    'max_num_seqs': 32,
    'gpu_memory_utilization': 0.85,
    'max_model_len': 8192,
    'dtype': 'auto',
    'trust_remote_code': True,
    'enable_prefix_caching': True,
    'enable_chunked_prefill': True,
    # max_lora_rank raised to 64 for the r/alpha grid search
    # (r=64 exceeds the competition ceiling of 32 -> off-competition only;
    #  64 also correctly evaluates r<=32 adapters, results unchanged).
    'max_lora_rank': 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-name',
        required=True,
        help='Short model identifier, used as the prefix for output files.',
    )
    parser.add_argument(
        '--base-model',
        required=True,
        help='Path to the local base model weights directory (vLLM `model=`).',
    )
    parser.add_argument(
        '--adapter',
        required=True,
        help='Path to the LoRA adapter directory (contains adapter_config.json '
             'and adapter_model.safetensors).',
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
    base_model_path = Path(args.base_model).resolve()
    if not base_model_path.exists():
        raise FileNotFoundError(f'base model path does not exist: {base_model_path}')

    adapter_path = Path(args.adapter).resolve()
    if not (adapter_path / 'adapter_config.json').exists():
        raise FileNotFoundError(
            f'adapter directory does not contain adapter_config.json: {adapter_path}'
        )

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

    provider_cfg = {
        'name': model_name,
        'model_path': str(base_model_path),
        'adapter_path': str(adapter_path),
        'provider': 'local_vllm',
        'chat': CHAT_KWARGS,
        'prompt_suffix': PROMPT_SUFFIX,
        'sampling': SAMPLING_PARAMS,
        'vllm': VLLM_ENGINE_PARAMS,
    }

    run_meta = {
        'model': model_name,
        'base_model': str(base_model_path),
        'adapter': str(adapter_path),
        'train_csv': args.train,
        'validation_ids': args.validation_ids,
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
            f'[evaluate_adapter] model={model_name} n={len(examples)} '
            f'base={base_model_path.name} adapter={adapter_path.name} '
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

    print(f'[evaluate_adapter] wrote {len(results)} records to {raw_path}')


if __name__ == '__main__':
    main()
