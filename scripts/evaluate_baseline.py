#!/usr/bin/env python3
"""Stage 2: score raw outputs produced by generate_baseline.py.

Reads `<model>_raw_outputs.jsonl` in each `--run-dir`, joins with the
validation split to recover gold answers and categories, applies the
shared answer extraction / verification helpers in
`src.evaluation.scoring` (which mirror the official NVIDIA metric
notebook), and writes per-model artifacts whose format matches
`notebooks/evaluation/adapter_validation.ipynb` cell 24:

    <model>_validation.csv     per-example: id, prompt, answer, output,
                               category, predicted, correct
    <model>_results.csv        per-category + TOTAL: correct, total,
                               weightage, percentage, contribution
    <model>_mistakes/<cat>.csv rows where correct=False, one file per
                               category

Usage:
    python scripts/evaluate_baseline.py \
        --run-dir results/prompting_baselines/nemotron-3-nano-30b-a3b \
        [--run-dir results/prompting_baselines/qwen3-30b-a3b ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.scoring import extract_final_answer, verify  # noqa: E402
from src.prompting.dataset import load_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-dir',
        required=True,
        action='append',
        help='Directory produced by generate_baseline.py. Repeatable.',
    )
    parser.add_argument(
        '--train',
        default='data/train.csv',
    )
    parser.add_argument(
        '--validation-ids',
        default='configs/eval/validation_ids_seed42_size950.csv',
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    if os.path.isabs(path):
        return Path(path)
    return (REPO_ROOT / path).resolve()


def _discover_run(run_dir: Path) -> tuple[str, Path]:
    """Locate `<model>_raw_outputs.jsonl` in run_dir and return (model, path)."""
    candidates = sorted(run_dir.glob('*_raw_outputs.jsonl'))
    if not candidates:
        raise FileNotFoundError(
            f'No *_raw_outputs.jsonl found in {run_dir}. '
            'Did you run scripts/generate_baseline.py first?'
        )
    if len(candidates) > 1:
        raise ValueError(
            f'Multiple *_raw_outputs.jsonl files in {run_dir}: {candidates!r}. '
            'Give each model its own --out directory.'
        )
    raw_path = candidates[0]
    model_name = raw_path.name[: -len('_raw_outputs.jsonl')]
    return model_name, raw_path


def evaluate_run(run_dir: Path, examples_by_id: dict) -> None:
    model_name, raw_path = _discover_run(run_dir)

    run_yaml_path = run_dir / f'{model_name}_run.yaml'
    run_meta: dict = {}
    if run_yaml_path.exists():
        with run_yaml_path.open('r', encoding='utf-8') as f:
            run_meta = yaml.safe_load(f) or {}

    # Load raw outputs and join with gold labels.
    rows: list[dict] = []
    missing_ids: list[str] = []
    with raw_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = obj['id']
            raw_output = obj.get('raw_output', '')
            ex = examples_by_id.get(row_id)
            if ex is None:
                missing_ids.append(row_id)
                continue
            predicted = extract_final_answer(raw_output)
            correct = verify(str(ex.answer), str(predicted))
            rows.append(
                {
                    'id': row_id,
                    'prompt': ex.prompt,
                    'answer': ex.answer,
                    'output': raw_output,
                    'category': ex.category,
                    'predicted': predicted,
                    'correct': bool(correct),
                }
            )

    if missing_ids:
        print(
            f'[evaluate] warning: {len(missing_ids)} raw_output ids not in '
            f'validation split: {missing_ids[:3]}...',
            file=sys.stderr,
        )

    df = pd.DataFrame(rows)

    # --- validation.csv: full per-example table (matches notebook cell 24) ---
    validation_path = run_dir / f'{model_name}_validation.csv'
    df.to_csv(validation_path, index=False)

    # --- mistakes per category (matches notebook cell 24) ---
    mistakes_dir = run_dir / f'{model_name}_mistakes'
    if mistakes_dir.exists():
        for old in mistakes_dir.glob('*.csv'):
            old.unlink()
    mistakes_dir.mkdir(parents=True, exist_ok=True)
    for category in df['category'].unique():
        cat_mistakes = df[(df['category'] == category) & (~df['correct'])]
        if not cat_mistakes.empty:
            cat_mistakes.to_csv(mistakes_dir / f'{category}.csv', index=False)

    # --- results.csv: per-category + TOTAL summary (matches notebook cell 24) ---
    stats = (
        df.groupby('category')['correct']
        .agg(correct='sum', total='count')
        .sort_index()
    )
    stats['correct'] = stats['correct'].astype('int')
    grand_total = int(stats['total'].sum())
    stats['weightage'] = (stats['total'] / grand_total * 100).map('{:.1f}%'.format)
    stats['percentage'] = (stats['correct'] / stats['total'] * 100).map('{:.1f}%'.format)
    stats['contribution'] = (stats['correct'] / grand_total * 100).map('{:.1f}%'.format)
    overall_correct = int(stats['correct'].sum())
    overall_pct = overall_correct / grand_total * 100 if grand_total else 0.0
    totals = pd.DataFrame(
        {
            'correct': [overall_correct],
            'total': [grand_total],
            'weightage': ['100.0%'],
            'percentage': [f'{overall_pct:.1f}%'],
            'contribution': [f'{overall_pct:.1f}%'],
        },
        index=['TOTAL'],
    )
    results = pd.concat([stats, totals])
    print(f'=== {model_name} ===')
    print(results.to_string())
    results.to_csv(run_dir / f'{model_name}_results.csv')

    # Small provenance breadcrumb so the run_yaml metadata is retained
    # alongside the CSVs. Not part of the original notebook output but
    # useful when several runs live side by side.
    if run_meta:
        print(
            f'[evaluate] {model_name}: {overall_correct}/{grand_total} '
            f'({overall_pct:.1f}%), git={run_meta.get("git_commit", "?")[:8]}'
        )


def main() -> None:
    args = parse_args()

    train_csv = resolve(args.train)
    ids_csv = resolve(args.validation_ids)
    examples = load_validation(train_csv, ids_csv)
    examples_by_id = {ex.id: ex for ex in examples}

    for run_dir in args.run_dir:
        evaluate_run(Path(run_dir).resolve(), examples_by_id)


if __name__ == '__main__':
    main()
