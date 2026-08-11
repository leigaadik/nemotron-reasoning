#!/usr/bin/env python3
"""Fabricate a low-quality CoT dataset from data/train.csv for an SFT ablation.

The generated reasoning is a template-filled restatement of the prompt's own
examples plus the already-known gold answer -- it contains no real derivation
step, unlike the legacy external CoT or the solver-verified synthetic CoT.
Output schema matches data/train_split_with_cot.csv (id,prompt,answer,type,
generated_cot) so it can be consumed by the existing `source: legacy` adapter
(src/data/legacy.py::iter_legacy_cot) without any pipeline changes.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


def _classify(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r"[01]{4,}", prompt):
        return "mixed" if any(w in p for w in ("word", "text", "letter", "phrase", "string")) else "binary"
    if any(w in p for w in ("word", "phrase", "text", "letter", "string", "character")):
        return "text"
    return "numeric"


def _cot(prompt: str, answer: str) -> str:
    bucket = _classify(prompt)
    examples = [line.strip() for line in prompt.split("\n") if "->" in line or "→" in line][:3]
    ex_block = "".join(f"  {e}\n" for e in examples)
    templates = {
        "binary": (
            "Let me analyse the binary transformation.\n\n"
            + (f"Examples:\n{ex_block}Comparing bits to find the rule.\n" if examples else "")
            + f"Applying the rule: {answer}"
        ),
        "text": (
            "Let me analyse the text transformation.\n\n"
            + (f"Examples:\n{ex_block}Building a substitution mapping.\n" if examples else "")
            + f"Applying the mapping: {answer}"
        ),
        "mixed": (
            "Let me decompose this mixed transformation.\n\n"
            + (f"Examples:\n{ex_block}Identifying separate rules per type.\n" if examples else "")
            + f"Combined result: {answer}"
        ),
        "numeric": (
            "Let me analyse the numeric pattern.\n\n"
            + (f"Examples:\n{ex_block}Checking arithmetic/conversion rules.\n" if examples else "")
            + f"Applying the pattern: {answer}"
        ),
    }
    return templates[bucket]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument("--output-csv", default="data/train_split_low_quality_cot.csv")
    parser.add_argument("--n", type=int, default=9500, help="Sample size (default matches legacy dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_path} missing columns: {sorted(missing)}")
        rows = list(reader)

    rng = random.Random(args.seed)
    if args.n < len(rows):
        rows = rng.sample(rows, args.n)
    else:
        rng.shuffle(rows)

    bucket_counts: dict[str, int] = {}
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
        writer.writeheader()
        for row in rows:
            prompt = row["prompt"]
            answer = row["answer"].strip()
            bucket = _classify(prompt)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            cot_text = _cot(prompt, answer)
            generated_cot = f"{cot_text}\n</think>\n\\boxed{{{answer}}}"
            writer.writerow(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "answer": answer,
                    "type": bucket,
                    "generated_cot": generated_cot,
                }
            )

    print(f"[done] wrote {len(rows)} rows to {output_path}")
    print(f"[buckets] {bucket_counts}")


if __name__ == "__main__":
    main()
