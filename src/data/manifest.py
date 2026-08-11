"""Dataset provenance helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(
    path: str | Path,
    *,
    stage: str,
    repo_root: Path,
    output_path: str | Path,
    details: dict[str, Any],
) -> None:
    path = Path(path)
    output_path = Path(output_path)
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
