#!/usr/bin/env python3
"""Combine the four evaluation summaries into one comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS = ("legacy-cot-transformers", "synthetic-cot-transformers")
SUITES = ("current_950",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for run_name in RUNS:
        for suite in SUITES:
            path = root / run_name / suite / f"{suite}_summary.csv"
            frame = pd.read_csv(path, index_col=0)
            total = frame.loc["TOTAL"]
            rows.append(
                {
                    "adapter": run_name,
                    "suite": suite,
                    "correct": int(total["correct"]),
                    "total": int(total["total"]),
                    "accuracy": float(total["accuracy"]),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(root / "matrix_summary.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
