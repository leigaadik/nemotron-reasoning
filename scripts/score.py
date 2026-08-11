#!/usr/bin/env python3
"""Score one raw-output JSONL against one configured benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reporting import score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--raw-outputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    score(
        suite_config=args.suite,
        raw_outputs=args.raw_outputs,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
