"""CoT generator for the physics_gravity question type.

Each problem gives several ``(t, d)`` observations obeying ``d = 0.5 * g * t^2``
under an unknown gravitational constant ``g``, then asks for the falling
distance at a query ``t``.

The CoT derives ``g`` from each example (``g_i = 2 * d_i / t_i^2``), takes the
median across examples, then applies ``d = 0.5 * g * t^2`` to the query ``t``.

New contract (see ``nemotron/data/main_router.py``):

    generate_cot(prompt: str, answer: str, **kwargs)
        -> (cot_text: str, meta: Dict[str, Any])

``meta`` always contains:

* ``predicted``: Optional[str] — genuine derivation from the prompt (never
  populated from ``answer``). Always a string for this category.
* ``correct``:  Optional[bool] — ``predicted == answer.strip()``.

Plus these diagnostic keys:

* ``g_values``: List[float] — per-example ``g_i``.
* ``g_median``: float — median ``g`` used for the final computation.
* ``query_t``: float — query time pulled from the prompt.
"""

import math
import re
import statistics
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_EXAMPLE_RE = re.compile(
    r"t\s*=\s*([\d.]+)\s*s,\s*distance\s*=\s*([\d.]+)\s*m"
)
_QUERY_RE = re.compile(r"falling distance for t\s*=\s*([\d.]+)\s*s")


def _parse_examples(prompt: str) -> List[Tuple[str, str]]:
    """Return list of ``(t_str, d_str)`` preserving the prompt's text form."""
    return [(m.group(1), m.group(2)) for m in _EXAMPLE_RE.finditer(prompt)]


def _parse_query(prompt: str) -> str:
    """Return the query ``t`` as it appears in the prompt (string form)."""
    m = _QUERY_RE.search(prompt)
    if not m:
        raise ValueError("Could not find query t in physics_gravity prompt.")
    return m.group(1)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_4sig(x: float) -> str:
    """Format ``x`` to 4 significant figures, preserving trailing zeros.

    e.g. 10.0 -> '10.00', 5.0 -> '5.000', 4.9 -> '4.900', 6.3014 -> '6.301',
    14.05 -> '14.05', 22.5 -> '22.50', 9.0 -> '9.000', 0.664 -> '0.6640'.
    """
    if x == 0:
        return "0.000"
    dp = max(0, 3 - int(math.floor(math.log10(abs(x)))))
    rounded = round(x, dp)
    # Recheck magnitude after rounding (handles e.g. 9.9995 -> '10.00').
    dp = max(0, 3 - int(math.floor(math.log10(abs(rounded)))))
    return f"{rounded:.{dp}f}"


# ---------------------------------------------------------------------------
# CoT rendering
# ---------------------------------------------------------------------------

def _render_example(
    idx: int, t_str: str, d_str: str, t: float, d: float
) -> List[str]:
    """Render the two-line ``Observation K:`` block."""
    two_d = _fmt_4sig(2.0 * d)
    t_sq = _fmt_4sig(t * t)
    g_i = 2.0 * d / (t * t)
    g_disp = _fmt_4sig(g_i)
    bridge = "So 2d" if idx == 1 else "2d"
    return [
        f"Observation {idx}",
        f"t={t_str}, d={d_str}. {bridge} = {two_d}. t^2 = {t_sq}."
        f" g = {two_d} / {t_sq} ≈ {g_disp}.",
    ]


def _render_tail(g_median: float, qt_str: str, qt: float) -> str:
    """Render the single-line final computation tail."""
    g_disp = _fmt_4sig(g_median)
    half_g_disp = _fmt_4sig(0.5 * g_median)
    qt_sq = _fmt_4sig(qt * qt)
    final_raw = 0.5 * g_median * qt * qt
    final_disp = _fmt_4sig(final_raw)
    return (
        f"Thus g ≈ {g_disp}."
        f" Compute for t={qt_str}. 0.5 * g = 0.5 * {g_disp} = {half_g_disp}."
        f" t^2 = {qt_str}^2 = {qt_sq}."
        f" So {half_g_disp} * {qt_sq} ≈ {final_disp}"
    )


def _render_cot(
    examples: List[Tuple[str, str]], qt_str: str
) -> Tuple[str, str, List[float], float, float]:
    """Build the CoT text and return it alongside diagnostic fields."""
    qt = float(qt_str)

    g_values: List[float] = []
    example_blocks: List[str] = []
    for i, (t_str, d_str) in enumerate(examples, start=1):
        t = float(t_str)
        d = float(d_str)
        g_i = 2.0 * d / (t * t)
        g_values.append(g_i)
        example_blocks.extend(_render_example(i, t_str, d_str, t, d))
        example_blocks.append("")

    if not g_values:
        raise ValueError("physics_gravity prompt contained no examples.")

    g_median = statistics.median(g_values)
    predicted = _fmt_4sig(0.5 * g_median * qt * qt)

    lines: List[str] = [
        "We need to find g from given data using d = 0.5 * g * t^2"
        " => g = 2d / t^2. "
        f"Use observations to compute g, then compute d for t = {qt_str}s. "
        "Use 4 significant figures.\n"
    ]
    lines.extend(example_blocks)
    lines.append(_render_tail(g_median, qt_str, qt))

    return "\n".join(lines), predicted, g_values, g_median, qt


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    examples = _parse_examples(prompt)
    qt_str = _parse_query(prompt)
    cot_text, predicted, g_values, g_median, query_t = _render_cot(examples, qt_str)
    return cot_text, {
        "predicted": predicted,
        "correct": predicted == answer.strip(),
        "g_values": g_values,
        "g_median": g_median,
        "query_t": query_t,
    }
