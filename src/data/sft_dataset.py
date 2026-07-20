"""Build SFT chat records from the cleaned CoT training CSV."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


@dataclass(frozen=True)
class SFTBuildResult:
    records: list[dict[str, Any]]
    labels: list[str]
    type_counts: dict[str, int]


def _require_columns(fieldnames: list[str], columns: list[str], csv_path: Path) -> None:
    available = set(fieldnames)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def clean_cot(cot: str) -> str:
    return BOXED_RE.sub("", cot).rstrip()


def build_sft_records(
    csv_path: str | Path,
    *,
    prompt_column: str,
    answer_column: str,
    cot_column: str,
    type_column: str,
    prompt_suffix: str,
    assistant_think_end_token: str,
    min_cot_chars: int,
    shuffle: bool,
    seed: int,
) -> SFTBuildResult:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    _require_columns(
        list(df.columns),
        [prompt_column, answer_column, cot_column, type_column],
        csv_path,
    )

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    labels: list[str] = []

    for _, row in df.iterrows():
        prompt = str(row[prompt_column])
        answer = str(row[answer_column])
        cot = str(row[cot_column])
        label = str(row[type_column])

        if not cot or cot == "nan" or len(cot.strip()) < min_cot_chars:
            continue

        cot_cleaned = clean_cot(cot)
        user_content = prompt + prompt_suffix
        assistant_content = (
            f"{cot_cleaned}\n{assistant_think_end_token}\n\\boxed{{{answer}}}"
        )

        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )
        labels.append(label)

    return SFTBuildResult(
        records=records,
        labels=labels,
        type_counts=dict(sorted(Counter(labels).items())),
    )


def to_hf_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset as HFDataset

    return HFDataset.from_list(records)
