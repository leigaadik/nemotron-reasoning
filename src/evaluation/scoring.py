"""Answer extraction and scoring utilities.

These helpers are adapted from the official NVIDIA Nemotron metric notebook and
kept dependency-free so they can be reused by scripts, notebooks, and tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from a model response.

    `\boxed{}` is the official target format, so it is prioritized. If no boxed
    answer is found, common final-answer phrases and then the last numeric value
    are used as fallbacks.
    """
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[index + 1].start() if index + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _strip_light_formatting(matches[-1])

    number_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if number_matches:
        return number_matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _strip_light_formatting(lines[-1]) if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Return whether a prediction matches a ground-truth answer.

    Binary strings are compared strictly. Other numeric answers allow the same
    tolerance used by the official metric. Non-numeric answers use
    case-insensitive exact matching.
    """
    stored_answer = _strip_light_formatting(str(stored_answer))
    predicted = _strip_light_formatting(str(predicted))

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def score_predictions(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Score prediction records.

    Each record must include `answer` and either `prediction` or `raw_output`.
    Optional `category` values are summarized separately.
    """
    total = 0
    correct = 0
    missing_boxed = 0
    by_category: dict[str, Counter[str]] = {}

    for record in records:
        total += 1
        raw_prediction = record.get("prediction")
        if raw_prediction is None:
            raw_prediction = extract_final_answer(record.get("raw_output"))
        prediction = str(raw_prediction)
        answer = str(record["answer"])
        is_correct = verify(answer, prediction)
        correct += int(is_correct)

        raw_output = record.get("raw_output")
        if raw_output is not None and r"\boxed{" not in str(raw_output):
            missing_boxed += 1

        category = str(record.get("category", "unknown"))
        if category not in by_category:
            by_category[category] = Counter()
        by_category[category]["total"] += 1
        by_category[category]["correct"] += int(is_correct)

    category_metrics = {
        category: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for category, counts in sorted(by_category.items())
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_boxed_count": missing_boxed,
        "categories": category_metrics,
    }


def _strip_light_formatting(value: str) -> str:
    value = value.strip()
    value = value.strip("`")
    value = value.rstrip(".")
    return value.strip()
