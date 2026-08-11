#!/usr/bin/env python3
"""Build a validated tokenized training dataset from one data config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.pipeline import build_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    print(json.dumps(build_data(config, config_path), indent=2))


if __name__ == "__main__":
    main()
