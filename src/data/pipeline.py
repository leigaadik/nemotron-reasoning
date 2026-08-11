"""Unified legacy and synthetic data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, resolve_path
from src.data.jsonl import read_jsonl, write_jsonl
from src.data.legacy import iter_legacy_cot
from src.data.manifest import sha256_file, write_manifest
from src.data.synthetic import generate_problems, iter_verified_cot
from src.data.traces import iter_tokenized_dataset
from src.data.validation import validate_tokenized_rows, write_validation_report


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    model = config["model"]
    return AutoTokenizer.from_pretrained(
        resolve_path(model["tokenizer_path"]),
        trust_remote_code=bool(model.get("trust_remote_code", True)),
        local_files_only=True,
    )


def build_data(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data = config["data"]
    source = data["source"]
    traces_path = resolve_path(data["traces_jsonl"])
    tokenizer = _load_tokenizer(config)
    max_length = int(data.get("max_length", 8192))
    trace_report: dict[str, Any] = {}

    if source == "legacy":
        source_path = resolve_path(data["input_csv"])
        source_rows = iter_legacy_cot(
            source_path,
            min_cot_chars=int(data.get("min_cot_chars", 5)),
        )
        input_details = {
            "input": str(source_path),
            "input_sha256": sha256_file(source_path),
        }
        require_correct = False
    elif source == "synthetic":
        problems_path = resolve_path(data["problems_jsonl"])
        cot_path = resolve_path(data["cot_jsonl"])
        problems = generate_problems(
            data["category_counts"],
            seed=int(data.get("seed", 42)),
            generator_kwargs=data.get("generator_kwargs"),
        )
        write_jsonl(problems_path, problems)
        cot_report: dict[str, Any] = {}
        cot_rows = iter_verified_cot(
            read_jsonl(problems_path),
            tokenizer=tokenizer,
            cot_kwargs=data.get("cot_kwargs"),
            report=cot_report,
        )
        write_jsonl(cot_path, cot_rows)
        input_details = {
            "problems": str(problems_path),
            "problems_sha256": sha256_file(problems_path),
            "cot": str(cot_path),
            "cot_sha256": sha256_file(cot_path),
            "cot_quality": cot_report,
        }
        source_rows = read_jsonl(cot_path)
        require_correct = True
    else:
        raise ValueError(f"Unsupported data source: {source!r}")

    traces = iter_tokenized_dataset(
        source_rows,
        tokenizer,
        max_length=max_length,
        require_correct_cot=require_correct,
        report=trace_report,
    )
    write_jsonl(traces_path, traces)
    validation = validate_tokenized_rows(
        read_jsonl(traces_path),
        max_length=max_length,
        overlap_policy=str(data.get("evaluation_overlap", "forbid")),
    )
    validation_path = traces_path.with_suffix(".validation.json")
    write_validation_report(validation_path, validation)
    if validation["error_count"]:
        raise ValueError(
            f"Training data failed validation with {validation['error_count']} errors; "
            f"see {validation_path}"
        )
    write_manifest(
        traces_path.with_suffix(".manifest.json"),
        stage="build_training_data",
        repo_root=REPO_ROOT,
        output_path=traces_path,
        details={
            "config": str(config_path),
            "source": source,
            **input_details,
            "trace_report": trace_report,
            "validation": validation,
        },
    )
    return {"source": source, "traces": str(traces_path), **trace_report, "validation": validation}
