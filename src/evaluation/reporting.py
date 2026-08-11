"""Unified scoring and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.evaluation.scoring import extract_final_answer, verify
from src.evaluation.suites import load_suite


def score(
    *,
    suite_config: str | Path,
    raw_outputs: str | Path,
    output_dir: str | Path,
) -> Path:
    suite = load_suite(suite_config)
    raw_outputs = resolve_path(raw_outputs)
    output_dir = resolve_path(output_dir)
    examples = {example.id: example for example in suite.examples}
    raw_rows = [json.loads(line) for line in raw_outputs.read_text().splitlines() if line]
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("Raw outputs contain duplicate IDs")
    missing = sorted(set(examples) - set(raw_by_id))
    extra = sorted(set(raw_by_id) - set(examples))
    if missing or extra:
        raise ValueError(
            f"Suite/raw ID mismatch: missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    rows = []
    for example in suite.examples:
        raw = raw_by_id[example.id]
        predicted = extract_final_answer(raw.get("raw_output", ""))
        rows.append(
            {
                "id": example.id,
                "prompt": example.prompt,
                "answer": example.answer,
                "output": raw.get("raw_output", ""),
                "category": example.category,
                "predicted": predicted,
                "correct": verify(example.answer, predicted),
                "finish_reason": raw.get("finish_reason", ""),
                "tokens_out": raw.get("tokens_out", 0),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{suite.name}_validation.csv", index=False)
    stats = frame.groupby("category")["correct"].agg(correct="sum", total="count")
    stats["correct"] = stats["correct"].astype(int)
    stats["accuracy"] = stats["correct"] / stats["total"]
    total_correct = int(stats["correct"].sum())
    total = int(stats["total"].sum())
    totals = pd.DataFrame(
        {"correct": [total_correct], "total": [total], "accuracy": [total_correct / total]},
        index=["TOTAL"],
    )
    stats = pd.concat([stats, totals])
    stats["correct"] = stats["correct"].astype(int)
    stats["total"] = stats["total"].astype(int)
    summary_path = output_dir / f"{suite.name}_summary.csv"
    stats.to_csv(summary_path)

    mistakes = output_dir / f"{suite.name}_mistakes"
    mistakes.mkdir(parents=True, exist_ok=True)
    for old in mistakes.glob("*.csv"):
        old.unlink()
    for category, group in frame[~frame["correct"]].groupby("category"):
        group.to_csv(mistakes / f"{category.replace('/', '__')}.csv", index=False)
    print(stats.to_string())
    print(f"[score] {suite.name}: {total_correct}/{total} ({total_correct / total:.1%})")
    return summary_path
