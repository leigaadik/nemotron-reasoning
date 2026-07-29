#!/usr/bin/env python3
"""Create a fixed validation split for prompting baseline experiments."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.categories import detect_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--metadata",
        default="data/train_split_with_cot.csv",
        help="Optional CSV containing id,type metadata for stratification",
    )
    parser.add_argument(
        "--output",
        default="configs/eval/validation_ids_seed42_size950.csv",
        help="Output CSV path with id,type columns",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--size", type=int, default=950, help="Validation size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)

    rows = load_train_rows(train_path)
    if args.size > len(rows):
        raise ValueError(f"Requested {args.size} rows from only {len(rows)} examples")

    id_to_type = load_metadata_types(metadata_path)
    examples = [
        {
            "id": row["id"],
            "type": id_to_type.get(row["id"]) or detect_category(row["prompt"]),
        }
        for row in rows
    ]
    selected = stratified_sample(examples, args.size, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type"])
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} validation ids to {output_path}")
    print_category_counts(selected)


def load_train_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata_types(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []) or "type" not in (reader.fieldnames or []):
            return {}
        return {row["id"]: row["type"] for row in reader if row.get("id")}


def stratified_sample(
    examples: list[dict[str, str]], sample_size: int, seed: int
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        by_type[example["type"]].append(example)

    total = len(examples)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for category, items in by_type.items():
        exact = len(items) * sample_size / total
        base = int(exact)
        quotas[category] = base
        remainders.append((exact - base, category))

    remaining = sample_size - sum(quotas.values())
    for _, category in sorted(remainders, reverse=True)[:remaining]:
        quotas[category] += 1

    selected: list[dict[str, str]] = []
    for category in sorted(by_type):
        items = list(by_type[category])
        rng.shuffle(items)
        selected.extend(items[: quotas[category]])

    rng.shuffle(selected)
    return selected


def print_category_counts(rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["type"]] += 1
    for category in sorted(counts):
        print(f"{category}: {counts[category]}")


if __name__ == "__main__":
    main()
