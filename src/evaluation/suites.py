"""Configuration-driven evaluation-suite loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.prompting.dataset import Example


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvaluationSuite:
    name: str
    examples: list[Example]
    config_path: Path


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_suite(config_path: str | Path) -> EvaluationSuite:
    config_path = _resolve(config_path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    suite = config.get("suite") or {}
    name = str(suite.get("name") or config_path.stem)
    questions_path = _resolve(suite["questions_csv"])
    ids_value = suite.get("ids_csv")
    category_column = str(suite.get("category_column", "category"))

    with questions_path.open(newline="", encoding="utf-8-sig") as stream:
        question_rows = list(csv.DictReader(stream))
    if ids_value:
        by_id = {row["id"]: row for row in question_rows}
        ids_path = _resolve(ids_value)
        with ids_path.open(newline="", encoding="utf-8-sig") as stream:
            selection = list(csv.DictReader(stream))
        examples = []
        missing = []
        for selected in selection:
            row_id = selected["id"]
            row = by_id.get(row_id)
            if row is None:
                missing.append(row_id)
                continue
            examples.append(
                Example(
                    id=row_id,
                    prompt=row["prompt"],
                    answer=row["answer"],
                    category=selected.get(category_column) or selected.get("type") or "unknown",
                )
            )
        if missing:
            raise ValueError(f"{len(missing)} suite IDs missing from {questions_path}: {missing[:5]}")
    else:
        examples = [
            Example(
                id=row["id"],
                prompt=row["prompt"],
                answer=row["answer"],
                category=row.get(category_column) or "unknown",
            )
            for row in question_rows
        ]

    expected = suite.get("expected_size")
    if expected is not None and len(examples) != int(expected):
        raise ValueError(
            f"Suite {name!r} expected {expected} examples, loaded {len(examples)}"
        )
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"Suite {name!r} contains duplicate IDs")
    return EvaluationSuite(name=name, examples=examples, config_path=config_path)
