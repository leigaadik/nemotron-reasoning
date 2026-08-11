"""Validation for the common tokenized training dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def load_evaluation_identity() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    with (REPO_ROOT / "data/train.csv").open(newline="", encoding="utf-8") as stream:
        train_rows = {row["id"]: row for row in csv.DictReader(stream)}
    with (REPO_ROOT / "configs/eval/validation_ids_seed42_size950.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            ids.add(row["id"])
            prompt_hashes.add(prompt_hash(train_rows[row["id"]]["prompt"]))
    return ids, prompt_hashes


def validate_tokenized_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_length: int,
    overlap_policy: str,
) -> dict[str, Any]:
    if overlap_policy not in {"allow", "forbid"}:
        raise ValueError(f"Invalid overlap policy: {overlap_policy!r}")
    eval_ids, eval_prompt_hashes = load_evaluation_identity()
    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()
    errors: list[str] = []
    overlap_ids = 0
    overlap_prompts = 0
    total_tokens = 0
    total_loss_tokens = 0
    row_count = 0

    for line_number, row in enumerate(rows, start=1):
        row_count += 1
        row_id = str(row.get("id", ""))
        input_ids = row.get("input_ids") or []
        labels = row.get("labels") or []
        prompt_tokens = int(row.get("num_prompt_tokens", -1))
        if not row_id or row_id in seen_ids:
            errors.append(f"line {line_number}: empty or duplicate ID {row_id!r}")
        seen_ids.add(row_id)
        id_overlap = row_id in eval_ids
        prompt_overlap = prompt_hash(str(row.get("prompt", ""))) in eval_prompt_hashes
        overlap_ids += int(id_overlap)
        overlap_prompts += int(prompt_overlap)
        if overlap_policy == "forbid" and (id_overlap or prompt_overlap):
            errors.append(f"line {line_number}: overlaps evaluation data: {row_id}")
        if not input_ids or len(input_ids) != len(labels):
            errors.append(f"line {line_number}: invalid input_ids/labels")
            continue
        if len(input_ids) > max_length:
            errors.append(f"line {line_number}: sequence exceeds {max_length}")
        if not 0 < prompt_tokens < len(labels):
            errors.append(f"line {line_number}: invalid prompt boundary")
        elif any(int(label) != -100 for label in labels[:prompt_tokens]):
            errors.append(f"line {line_number}: prompt labels are not masked")
        if all(int(label) == -100 for label in labels):
            errors.append(f"line {line_number}: no loss-bearing labels")
        categories[str(row.get("category", "unknown"))] += 1
        total_tokens += len(input_ids)
        total_loss_tokens += sum(int(label) != -100 for label in labels)

    if row_count == 0:
        errors.append("dataset is empty")
    return {
        "rows": row_count,
        "categories": dict(sorted(categories.items())),
        "total_tokens": total_tokens,
        "total_loss_tokens": total_loss_tokens,
        "evaluation_overlap": {
            "policy": overlap_policy,
            "id_rows": overlap_ids,
            "prompt_rows": overlap_prompts,
        },
        "error_count": len(errors),
        "errors": errors[:100],
    }


def write_validation_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
