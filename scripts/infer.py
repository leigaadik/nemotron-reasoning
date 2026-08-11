#!/usr/bin/env python3
"""Run base-model or LoRA inference on one configured benchmark suite."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure spawned vLLM engine processes also avoid the platform's ABI-broken
# optional CUDA extensions.
compat_path = REPO_ROOT / "scripts" / "eval_compat"
os.environ["NEMOTRON_EVAL_DISABLE_BROKEN_EXTENSIONS"] = "1"
os.environ["PYTHONPATH"] = (
    str(compat_path)
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)
sys.modules.setdefault("flash_attn", None)
sys.modules.setdefault("fbgemm_gpu", None)

from src.inference.runner import infer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    infer(
        run_name=args.run_name,
        base_model=args.base_model,
        adapter=args.adapter,
        suite_config=args.suite,
        output_dir=args.out,
        force=args.force,
    )


if __name__ == "__main__":
    main()
