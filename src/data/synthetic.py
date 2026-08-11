"""Adapters around the vendored reference synthetic generators and solvers."""

from __future__ import annotations

import inspect
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.scoring import verify


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "src" / "generators"
REFERENCE_COMMIT = "597dcaa"


def _load_reference_modules():
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from nemotron.data.main_router import (  # type: ignore
        COT_GENERATORS,
        SYN_GENERATORS,
        resolve_category,
    )

    return COT_GENERATORS, SYN_GENERATORS, resolve_category


def generate_problems(
    category_counts: dict[str, int],
    *,
    seed: int,
    generator_kwargs: dict[str, dict[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield deterministic problem rows using the reference category generators."""
    _, generators, _ = _load_reference_modules()
    generator_kwargs = generator_kwargs or {}
    for offset, category in enumerate(sorted(category_counts)):
        count = int(category_counts[category])
        if count < 0:
            raise ValueError(f"Negative count for {category}: {count}")
        if category not in generators:
            raise ValueError(
                f"Unknown synthetic category {category!r}; available={sorted(generators)}"
            )
        cls = generators[category]
        requested = generator_kwargs.get(category, {})
        accepted = inspect.signature(cls.__init__).parameters
        kwargs = {key: value for key, value in requested.items() if key in accepted}
        generator = cls(seed=seed + offset, **kwargs)
        for index, row in enumerate(generator.generate(count)):
            content_hash = hashlib.sha256(
                (str(row["prompt"]) + "\0" + str(row["answer"])).encode("utf-8")
            ).hexdigest()[:16]
            yield {
                "id": (
                    f"syn-{category.replace('/', '-')}-{seed + offset}-"
                    f"{index:06d}-{content_hash}"
                ),
                "source": "reference_synthetic",
                "category": category,
                "prompt": row["prompt"],
                "gold_answer": row["answer"],
                "generator_seed": seed + offset,
                "reference_commit": REFERENCE_COMMIT,
            }


def generate_verified_cot(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any = None,
    cot_kwargs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate solver CoT and retain explicit measurable/correct quality state."""
    report: dict[str, Any] = {}
    output = list(
        iter_verified_cot(
            rows,
            tokenizer=tokenizer,
            cot_kwargs=cot_kwargs,
            report=report,
        )
    )
    return output, report


def iter_verified_cot(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any = None,
    cot_kwargs: dict[str, dict[str, Any]] | None = None,
    report: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    """Stream verified CoT rows while updating ``report`` in place."""
    cot_generators, _, resolve_category = _load_reference_modules()
    cot_kwargs = cot_kwargs or {}
    states: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = {}
    total = 0

    for source_row in rows:
        row = dict(source_row)
        category = str(row["category"])
        if category not in cot_generators:
            raise ValueError(f"No CoT solver for category {category!r}")
        solver_row = {
            "id": row["id"],
            "category": category,
            "prompt": row["prompt"],
            "answer": row["gold_answer"],
        }
        resolved = resolve_category(solver_row)
        kwargs = dict(cot_kwargs.get(category) or {})
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        try:
            cot, metadata = cot_generators[resolved].generate_cot(
                solver_row["prompt"], solver_row["answer"], **kwargs
            )
            derived = metadata.get("predicted")
            measurable = derived is not None
            correct = measurable and verify(str(row["gold_answer"]), str(derived))
            error = None
        except Exception as exc:
            cot, metadata, derived = "", {}, None
            measurable, correct = False, False
            error = f"{type(exc).__name__}: {exc}"

        state = "correct" if correct else "incorrect" if measurable else "unmeasurable"
        states[state] += 1
        per_category.setdefault(category, Counter())[state] += 1
        total += 1
        row.update(
            {
                "cot": cot,
                "derived_answer": derived,
                "cot_measurable": measurable,
                "cot_correct": correct,
                "cot_quality": "solver_verified",
                "cot_state": state,
                "cot_error": error,
                "cot_metadata": metadata,
            }
        )
        yield row

    if report is not None:
        report.update(
            {
                "total": total,
                "states": dict(sorted(states.items())),
                "categories": {
                    category: dict(sorted(counts.items()))
                    for category, counts in sorted(per_category.items())
                },
            }
        )
