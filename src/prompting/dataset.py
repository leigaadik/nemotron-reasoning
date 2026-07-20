"""Load the fixed validation split for prompting baseline experiments.

Joins `data/train.csv` (id -> prompt, answer) with the pre-computed
validation ID list at `configs/eval/validation_ids_seed42_size950.csv`
(id -> category). The category column already reflects the stratified
sample built by `scripts/create_validation_split.py`, so it is reused
verbatim rather than re-inferred with `detect_category`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    answer: str
    category: str


def load_validation(train_csv: Path, ids_csv: Path) -> list[Example]:
    train_csv = Path(train_csv)
    ids_csv = Path(ids_csv)

    with train_csv.open(newline='', encoding='utf-8') as f:
        train = {row['id']: row for row in csv.DictReader(f)}

    with ids_csv.open(newline='', encoding='utf-8') as f:
        ids_rows = list(csv.DictReader(f))

    examples: list[Example] = []
    missing: list[str] = []
    for row in ids_rows:
        row_id = row['id']
        if row_id not in train:
            missing.append(row_id)
            continue
        train_row = train[row_id]
        examples.append(
            Example(
                id=row_id,
                prompt=train_row['prompt'],
                answer=train_row['answer'],
                category=row.get('type') or 'unknown',
            )
        )

    if missing:
        raise ValueError(
            f'{len(missing)} validation ids missing from {train_csv}: '
            f'{missing[:5]}{"..." if len(missing) > 5 else ""}'
        )
    return examples
