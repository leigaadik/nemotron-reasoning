"""Legacy external-CoT dataset adapter."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterator


BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def clean_external_cot(cot: str) -> str:
    reasoning = cot.split("</think>", 1)[0]
    return BOXED_RE.sub("", reasoning).rstrip()


def iter_legacy_cot(
    csv_path: str | Path,
    *,
    prompt_column: str = "prompt",
    answer_column: str = "answer",
    cot_column: str = "generated_cot",
    category_column: str = "type",
    min_cot_chars: int = 5,
) -> Iterator[dict[str, Any]]:
    """Convert the historical CSV into the common pre-tokenized schema."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "id",
            prompt_column,
            answer_column,
            cot_column,
            category_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        for row in reader:
            cot = str(row[cot_column]).strip()
            if not cot or cot == "nan" or len(cot) < min_cot_chars:
                continue
            answer = str(row[answer_column]).strip()
            yield {
                "id": str(row["id"]),
                "source": "legacy_external_cot",
                "category": str(row[category_column]),
                "prompt": str(row[prompt_column]),
                "gold_answer": answer,
                "target_answer": answer,
                "derived_answer": None,
                "cot": clean_external_cot(cot),
                "cot_quality": "external_unverified",
                "cot_correct": None,
            }
