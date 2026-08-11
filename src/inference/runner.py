"""Unified vLLM inference for base-model and LoRA evaluation runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import REPO_ROOT, resolve_path
from src.evaluation.suites import load_suite
from src.providers import build_provider


PROMPT_SUFFIX = (
    'Please put your final answer inside `\\boxed{}`. '
    'For example: `\\boxed{your answer}`'
)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 7680}
VLLM = {
    "tensor_parallel_size": 1,
    "max_num_seqs": 32,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "dtype": "auto",
    "trust_remote_code": True,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "max_lora_rank": 32,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def infer(
    *,
    run_name: str,
    base_model: str | Path,
    suite_config: str | Path,
    output_dir: str | Path,
    adapter: str | Path | None = None,
    force: bool = False,
) -> Path:
    suite = load_suite(suite_config)
    base_model = resolve_path(base_model)
    adapter_path = resolve_path(adapter) if adapter else None
    output_dir = resolve_path(output_dir)
    if not base_model.exists():
        raise FileNotFoundError(f"Base model does not exist: {base_model}")
    if adapter_path and not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Invalid adapter: {adapter_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}__{suite.name}"
    raw_path = output_dir / f"{prefix}_raw_outputs.jsonl"
    if raw_path.exists() and not force:
        raise FileExistsError(f"{raw_path} exists; pass --force to overwrite")

    provider_config: dict[str, Any] = {
        "name": run_name,
        "model_path": str(base_model),
        "provider": "local_vllm",
        "chat": {"enable_thinking": True, "system": None},
        "sampling": SAMPLING,
        "vllm": VLLM,
    }
    if adapter_path:
        provider_config["adapter_path"] = str(adapter_path)
    metadata = {
        "run_name": run_name,
        "suite": suite.name,
        "suite_config": str(suite.config_path),
        "base_model": str(base_model),
        "adapter": str(adapter_path) if adapter_path else None,
        "num_examples": len(suite.examples),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "params": {
            "prompt_suffix": PROMPT_SUFFIX,
            "sampling": SAMPLING,
            "vllm": VLLM,
        },
    }
    (output_dir / f"{prefix}_run.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )

    provider = build_provider(provider_config)
    try:
        results = provider.generate(suite.examples, PROMPT_SUFFIX)
    finally:
        provider.close()
    with raw_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(
                json.dumps(
                    {
                        "id": result.id,
                        "prompt": result.prompt,
                        "raw_output": result.raw_output,
                        "finish_reason": result.finish_reason,
                        "tokens_out": result.tokens_out,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[infer] {run_name}/{suite.name}: wrote {len(results)} rows to {raw_path}")
    return raw_path
