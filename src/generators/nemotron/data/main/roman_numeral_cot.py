"""CoT generator for the roman_numeral question type.

Each problem gives several integer -> Roman numeral examples under the standard
system, then asks the model to convert a query integer.

New contract (see ``nemotron/data/main_router.py``):

    generate_cot(prompt: str, answer: str, **kwargs)
        -> (cot_text: str, meta: Dict[str, Any])

``meta`` always contains:

* ``predicted``: Optional[str] — genuine derivation from the prompt (never
  populated from ``answer``). Always a string for this category.
* ``correct``:  Optional[bool] — ``predicted == answer.strip()``.
"""

import re
from typing import Any, Dict, List, Tuple


_TENS: List[Tuple[int, str]] = [
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
]

_ONES: List[Tuple[int, str]] = [
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _place_roman(n: int, table: List[Tuple[int, str]]) -> str:
    result = ""
    for val, sym in table:
        while n >= val:
            result += sym
            n -= val
    return result


def _decompose(n: int) -> List[Tuple[int, str]]:
    hundreds = (n // 100) * 100
    tens = ((n % 100) // 10) * 10
    ones = n % 10
    parts: List[Tuple[int, str]] = []
    for place, table in [(hundreds, _TENS), (tens, _TENS), (ones, _ONES)]:
        if place:
            parts.append((place, _place_roman(place, table)))
    return parts


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_query(prompt: str) -> int:
    m = re.search(r"write the number\s+(\d+)", prompt, re.IGNORECASE)
    if not m:
        raise ValueError("Could not find query integer in roman_numeral prompt.")
    return int(m.group(1))


def _parse_examples(prompt: str) -> List[Tuple[int, str]]:
    return [
        (int(m.group(1)), m.group(2))
        for m in re.finditer(r"(\d+)\s*->\s*([A-Z]+)", prompt)
    ]


# ---------------------------------------------------------------------------
# CoT rendering
# ---------------------------------------------------------------------------

def _render_example(v: int, r: str) -> str:
    """Format one `Example K` block body.

    Single-place values (e.g. 10, 100) get a terse ``10 -> X.`` line; multi-place
    values get an expansion: ``28 -> XXVIII. 20 -> XX, 8 -> VIII, so 28 -> XXVIII.``
    """
    parts = _decompose(v)
    if len(parts) <= 1:
        return f"{v} -> {r}."
    place_str = ", ".join(f"{pv} -> {pr}" for pv, pr in parts)
    return f"{v} -> {r}. {place_str}, so {v} -> {r}."


def _render_convert(n: int, r: str) -> str:
    """Format the final ``Convert N. ...`` tail.

    Single-place: ``Convert 8. 8 -> VIII.``
    Multi-place:  ``Convert 83. 80 -> LXXX, 3 -> III, so 83 -> LXXXIII.``
    """
    parts = _decompose(n)
    if len(parts) <= 1:
        return f"Convert {n}. {n} -> {r}."
    place_str = ", ".join(f"{pv} -> {pr}" for pv, pr in parts)
    return f"Convert {n}. {place_str}, so {n} -> {r}"


def _render_cot(n: int, examples: List[Tuple[int, str]]) -> Tuple[str, str]:
    predicted = "".join(r for _, r in _decompose(n))

    lines: List[str] = [
        "We need to infer the numeral system used. Let's examine examples.",
        "",
    ]
    for i, (v, r) in enumerate(examples, start=1):
        lines.append(f"Example {i}")
        lines.append(_render_example(v, r))
        lines.append("")
    lines.append(
        f"So it's Roman numerals. {_render_convert(n, predicted)}"
    )
    return "\n".join(lines), predicted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    n = _parse_query(prompt)
    examples = _parse_examples(prompt)
    cot_text, predicted = _render_cot(n, examples)
    return cot_text, {
        "predicted": predicted,
        "correct": predicted == answer.strip(),
    }
