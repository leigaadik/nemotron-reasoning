"""CoT generator for the unit_conversion question type.

Each problem presents several ``X m becomes Y`` example lines (where X is the
raw/input measurement and Y is the converted output) and asks the model to
convert a query measurement.  The secret conversion is a single scalar
``k ~ Uniform(0.5, 2.0)`` such that ``output = round(k * input, 2)``.

New contract (see ``nemotron/data/main_router.py``):

    generate_cot(prompt: str, answer: str, **kwargs)
        -> (cot_text: str, meta: Dict[str, Any])

``meta`` always contains:

* ``predicted``: Optional[str] — genuine derivation from the prompt (never
  populated from ``answer``). Always a string for this category.
* ``correct``:  Optional[bool] — ``predicted == answer.strip()``.

Plus diagnostic keys: ``k_list`` (per-example 4-sig-fig strings),
``k_median`` (the chosen median k as 4-sig-fig string), and ``x_query``
(the query input string from the prompt).

Sample output (body between <think>...</think>):

    We need to find the conversion rule. Let's compute examples with 4 significant figures.

    Example 1:
    44.49 / 28.98 ≈ 1.535.

    Example 2:
    26.30 / 17.13 ≈ 1.535.

    Thus the median factor is 1.535. Computing for 20.97: 20.97 * 1.535 ≈ 32.19.
"""

import math
import re
import statistics
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_examples(prompt: str) -> List[Tuple[str, str]]:
    """Return list of ``(input_str, output_str)`` pairs as textual copies.

    In the prompt ``X m becomes Y``, X is the raw input and Y is the
    converted output.  Strings are preserved verbatim so the CoT can echo
    them exactly as they appear.
    """
    return [
        (m.group(1), m.group(2))
        for m in re.finditer(r"([\d.]+)\s+\w+\s+becomes\s+([\d.]+)", prompt)
    ]


def _parse_query(prompt: str) -> str:
    m = re.search(r"convert the following measurement[:\s]+([\.\d]+)", prompt)
    if not m:
        raise ValueError("Could not find query input in unit_conversion prompt.")
    return m.group(1)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_4sig(x: float) -> str:
    """Format ``x`` to 4 significant figures, preserving trailing zeros.

    e.g. 10.0 -> '10.00', 5.0 -> '5.000', 4.9 -> '4.900', 1.535 -> '1.535',
    0.664 -> '0.6640'.
    """
    if x == 0:
        return "0.000"
    dp = max(0, 3 - int(math.floor(math.log10(abs(x)))))
    rounded = round(x, dp)
    dp = max(0, 3 - int(math.floor(math.log10(abs(rounded)))))
    return f"{rounded:.{dp}f}"


# ---------------------------------------------------------------------------
# CoT rendering
# ---------------------------------------------------------------------------

def _render_cot(
    examples: List[Tuple[str, str]],
    query_in_str: str,
    predicted_str: str,
    k_str: str,
    k_list_str: List[str],
) -> str:
    lines: List[str] = [
        "We need to infer the conversion rule. Let's compute examples with 4 significant figures.",
        "",
    ]
    for i, ((in_str, out_str), ki_str) in enumerate(zip(examples, k_list_str), start=1):
        lines.append(f"Example {i}")
        lines.append(f"{in_str} m -> {out_str}. Ratio approx {out_str} / {in_str} ≈ {ki_str}.")
        lines.append("")
    lines.append(
        f"Thus factor ≈ {k_str}. "
        f"Compute for {query_in_str}. {query_in_str} * {k_str} ≈ {predicted_str}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    examples = _parse_examples(prompt)
    query_in_str = _parse_query(prompt)

    # Per-example k = output / input (both from the prompt's textual pairs).
    k_values: List[float] = [float(out) / float(inp) for inp, out in examples]
    k_list_str: List[str] = [_fmt_4sig(k) for k in k_values]

    k_median = statistics.median(k_values)
    k_str = _fmt_4sig(k_median)

    # Predicted answer uses the 4-sig-fig-rounded median k (the value the CoT
    # actually shows) to stay self-consistent with the rendered trace.
    x_query = float(query_in_str)
    predicted_str = _fmt_4sig(float(k_str) * x_query)

    cot_text = _render_cot(examples, query_in_str, predicted_str, k_str, k_list_str)

    return cot_text, {
        "predicted": predicted_str,
        "correct": predicted_str == answer.strip(),
        "k_list": k_list_str,
        "k_median": k_str,
        "x_query": query_in_str,
    }
