"""Qwen chat rendering, completion-only labels, and offline packing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


IGNORE_INDEX = -100
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def _encode(tokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return list(tokens)


def build_tokenized_trace(
    row: dict[str, Any],
    tokenizer,
    *,
    max_length: int,
    require_correct_cot: bool = True,
) -> dict[str, Any]:
    """Build one Qwen sequence with prompt tokens masked from the loss."""
    if require_correct_cot and not row.get("cot_correct"):
        raise ValueError(f"Row {row.get('id')} does not have a verified correct CoT")
    cot = str(row.get("cot") or "").strip()
    if not cot:
        raise ValueError(f"Row {row.get('id')} has empty CoT")
    derived_answer = row.get("derived_answer")
    answer = str(derived_answer if derived_answer is not None else row["gold_answer"]).strip()
    user = str(row["prompt"]) + PROMPT_SUFFIX
    # Exact single-turn Qwen3 chat rendering from tokenizer_config.json. The
    # direct renderer keeps trace generation independent of Jinja versions and
    # makes the completion boundary explicit and auditable.
    prompt_text = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    completion_text = (
        f"<think>\n{cot}\n</think>\n\n"
        f"\\boxed{{{answer}}}<|im_end|>\n"
    )
    prompt_ids = _encode(tokenizer, prompt_text)
    input_ids = _encode(tokenizer, prompt_text + completion_text)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Chat-template prefix mismatch for {row.get('id')}: "
            f"prompt={len(prompt_ids)} full={len(input_ids)}"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if len(input_ids) != len(labels):
        raise AssertionError("input_ids/labels length mismatch")
    if len(input_ids) > max_length:
        raise ValueError(
            f"Row {row.get('id')} has {len(input_ids)} tokens, exceeds {max_length}"
        )
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError(f"Row {row.get('id')} has no completion loss tokens")

    return {
        "id": row["id"],
        "source": row.get("source", "unknown"),
        "category": row["category"],
        "prompt": row["prompt"],
        "gold_answer": str(row["gold_answer"]).strip(),
        "target_answer": answer,
        "derived_answer": derived_answer,
        "cot_quality": row.get("cot_quality", "solver_verified"),
        "cot_correct": row.get("cot_correct"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "num_tokens": len(input_ids),
        "num_prompt_tokens": len(prompt_ids),
        "num_loss_tokens": sum(label != IGNORE_INDEX for label in labels),
    }


def build_tokenized_dataset(
    rows: Iterable[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    require_correct_cot: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {}
    output = list(
        iter_tokenized_dataset(
            rows,
            tokenizer,
            max_length=max_length,
            require_correct_cot=require_correct_cot,
            report=report,
        )
    )
    return output, report


def iter_tokenized_dataset(
    rows: Iterable[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    require_correct_cot: bool = True,
    report: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    """Stream tokenized rows while updating aggregate statistics in place."""
    dropped: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    input_rows = 0
    written_rows = 0
    total_tokens = 0
    total_loss_tokens = 0
    for row in rows:
        input_rows += 1
        try:
            trace = build_tokenized_trace(
                row,
                tokenizer,
                max_length=max_length,
                require_correct_cot=require_correct_cot,
            )
        except ValueError as exc:
            reason = "too_long" if "exceeds" in str(exc) else "invalid"
            dropped[reason] += 1
            continue
        written_rows += 1
        categories[str(trace["category"])] += 1
        total_tokens += int(trace["num_tokens"])
        total_loss_tokens += int(trace["num_loss_tokens"])
        yield trace
    if report is not None:
        report.update(
            {
                "input_rows": input_rows,
                "written_rows": written_rows,
                "dropped": dict(sorted(dropped.items())),
                "categories": dict(sorted(categories.items())),
                "total_tokens": total_tokens,
                "total_loss_tokens": total_loss_tokens,
            }
        )
