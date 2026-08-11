"""CoT generator for equation_transform/numeric.

Puzzle: 3-5 examples ``a[op]b = c`` plus a query ``q_a[q_op]q_b``. The
catalog is 11 non-concat ops x {normal, reversed} + 2 concat ops (no
domain) = 24 rules. Reversed domain = digit-reverse operands, apply rule,
digit-reverse output.

Per-symbol state: ``(non_concat_normal, non_concat_reversed, concat)``.
Per-example pruning order:
  1. Reversed: non-concat rules on digit-reversed operands.
  2. Normal: non-concat rules on raw operands. Preceded by a reversal-back
     preamble that maps R(a)/R(b)/R(target) back to a/b/target.
  3. Concat: ``cat(a,b)`` -> a||b, ``cat(b,a)`` -> b||a, both on raw
     operands, compared against the raw target.
  4. Operand-zero filter: leading zero -> drop normal non-concat; trailing
     zero -> drop reversed non-concat. Concat is domain-free and never
     dropped here.

Five filters narrow the query symbol's survivors after all examples:
  f1 (cross-op domain). Each op votes normal/reversed iff its non-concat
     survivors are single-domain, else abstains. Concat survivors don't
     vote. Non-None votes must agree (asserted); consensus pins q_op's
     non-concat domain. Concat passes through.
  f2 (query-operand zero). Same rule as per-example, applied to q_a/q_b.
     Default to reversed when both f1 and f2 leave non-concat ambiguous.
     Concat passes through.
  f3 (bucket coverage). Buckets {sub, add, mul, concat}; only
     {sub, add, mul} are takeable. Each non-q_op takes its highest-
     priority takeable bucket. Drop non-concat q_op survivors whose
     bucket is taken; concat never drops. The ``max%min`` rule lives in
     the subtraction bucket (the dataset never pairs mod with other
     sub-family rules in examples), so a mod-only example op claims the
     entire subtraction bucket. If all drop, fall back to ``a-b``.
  f4 (subtraction-direction prune from display mode). The dataset encodes
     subtraction direction via the negative-output marker position:
     suffix ``Nop`` -> ``b-a`` direction; prefix ``opN`` or no negative
     example -> ``a-b`` direction. Drop the opposite-direction signed
     sub rule (``a-b`` on suffix, ``b-a`` otherwise) from query survivors.
     Breaks the otherwise-undecidable ``b-a | -|a-b|`` and
     ``a-b | -|a-b|`` plan-order tiebreaks.
  f5 (symbol affinity). If q_op is one of the standard arithmetic glyphs
     ``+``, ``-``, ``*``, restrict survivors to its family bucket
     (addition/subtraction/multiplication, incl. ±1 variants). No-op for
     other symbols. Falls back to keeping all f4 survivors if the
     intended family was already pruned out.

Pick: first plan-order survivor among non-concat normal, then non-concat
reversed, then concat.

generate_cot(prompt, answer, **kwargs) -> (cot, meta). meta keys:
predicted, correct, query_op, query_op_examples, chosen_rule,
chosen_domain, chosen_bucket, is_guess, display_mode.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Rule catalog -- 13 base ops, fixed display order
# ---------------------------------------------------------------------------

# (rule_id, display_name, bucket)
_OP_ORDER: List[Tuple[str, str, str]] = [
    ("a-b",      "subtraction without negation and without absolute value", "subtraction"),
    ("b-a",      "subtraction with negation and without absolute value",    "subtraction"),
    ("|a-b|",    "subtraction without negation and with absolute value",    "subtraction"),
    ("-|a-b|",   "subtraction with negation and with absolute value",       "subtraction"),
    ("max%min",  "subtraction through modulo",                    "subtraction"),
    ("a+b",      "addition",                                       "addition"),
    ("a+b+1",    "addition + 1",                                   "addition"),
    ("a+b-1",    "addition - 1",                                   "addition"),
    ("a*b",      "multiplication",                                 "multiplication"),
    ("a*b+1",    "multiplication + 1",                             "multiplication"),
    ("a*b-1",    "multiplication - 1",                             "multiplication"),
    ("cat(a,b)", "concatenation",                                  "concatenation"),
    ("cat(b,a)", "concatenation reversed",                          "concatenation"),
]

_RULE_IDS: List[str] = [r[0] for r in _OP_ORDER]
_DISPLAY_NAME: Dict[str, str] = {r[0]: r[1] for r in _OP_ORDER}
_BUCKET: Dict[str, str] = {r[0]: r[2] for r in _OP_ORDER}
_CAT_RULES = {"cat(a,b)", "cat(b,a)"}
_NON_CONCAT_RULE_IDS: List[str] = [r for r in _RULE_IDS if r not in _CAT_RULES]
_CONCAT_RULE_IDS: List[str] = [r for r in _RULE_IDS if r in _CAT_RULES]


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def _compute_numeric(rule_id: str, a: int, b: int) -> Optional[int]:
    """Evaluate a non-cat rule on (a, b). Returns int or None (bad mod)."""
    if rule_id == "a-b":      return a - b
    if rule_id == "|a-b|":    return abs(a - b)
    if rule_id == "b-a":      return b - a
    if rule_id == "-|a-b|":   return -abs(a - b)
    if rule_id == "a+b":      return a + b
    if rule_id == "a+b+1":    return a + b + 1
    if rule_id == "a+b-1":    return a + b - 1
    if rule_id == "a*b":      return a * b
    if rule_id == "a*b+1":    return a * b + 1
    if rule_id == "a*b-1":    return a * b - 1
    if rule_id == "max%min":
        if min(a, b) == 0:
            return None
        return max(a, b) % min(a, b)
    return None


def _compute_cat(rule_id: str, a_str: str, b_str: str) -> str:
    L = max(len(a_str), len(b_str), 2)
    a_pad = a_str.zfill(L)
    b_pad = b_str.zfill(L)
    if rule_id == "cat(a,b)":
        return a_pad + b_pad
    if rule_id == "cat(b,a)":
        return b_pad + a_pad
    return ""


def _reverse_digits(n: int) -> int:
    """Reverse digits of abs(n); preserve sign."""
    rev = str(abs(n))[::-1]
    return -int(rev) if n < 0 else int(rev)


def _sr_str(s: str, L: int) -> str:
    return s.zfill(L)[::-1]


def _domain_map_phrase(domain_word: str, pairs: List[Tuple[str, str]]) -> str:
    """Render ``'<src> <domain_word> is <dst>, ...'`` for each (src, dst).

    Shared by the reversed-section header of :func:`_render_example` and the
    operand-mapping opener of :func:`_final_deduction` so both phrase digit
    transforms identically (e.g. ``"45 reversed is 54, 54 reversed is 45"``).
    """
    return ", ".join(f"{src} {domain_word} is {dst}" for src, dst in pairs)


def _reversed_target_value(expected_digits: str, expected_is_neg: bool) -> int:
    """Reverse the digit string of the expected output, preserving sign.

    Unlike :func:`_reverse_digits` applied to an int, this preserves leading
    zeros (e.g. ``"0234" -> 4320`` rather than ``234 -> 432``).
    """
    rev = expected_digits[::-1]
    v = int(rev) if rev else 0
    return -v if expected_is_neg else v


def _matches_normal(
    rule_id: str, a_str: str, b_str: str,
    expected_val: int, expected_digits: str, expected_is_neg: bool,
) -> bool:
    if rule_id in _CAT_RULES:
        if expected_is_neg:
            return False
        return _compute_cat(rule_id, a_str, b_str) == expected_digits
    result = _compute_numeric(rule_id, int(a_str), int(b_str))
    if result is None:
        return False
    # Compare via string form to preserve leading-zero distinctions
    # (e.g. "0234" should not match a rule producing 234).
    if expected_is_neg:
        if result >= 0:
            return False
        return str(-result) == expected_digits
    if result < 0:
        return False
    return str(result) == expected_digits


def _matches_reversed(
    rule_id: str, a_str: str, b_str: str,
    expected_val: int, expected_digits: str, expected_is_neg: bool,
) -> bool:
    L = max(len(a_str), len(b_str), 2)
    sra_str = _sr_str(a_str, L)
    srb_str = _sr_str(b_str, L)
    if rule_id in _CAT_RULES:
        if expected_is_neg:
            return False
        cat_result = _compute_cat(rule_id, sra_str, srb_str)
        return cat_result[::-1] == expected_digits
    result = _compute_numeric(rule_id, int(sra_str), int(srb_str))
    if result is None:
        return False
    # Rule produces _R_str(result); compare that string form to expected.
    if expected_is_neg:
        if result >= 0:
            return False
        return str(-result)[::-1] == expected_digits
    if result < 0:
        return False
    return str(result)[::-1] == expected_digits


# ---------------------------------------------------------------------------
# Operand-zero heuristic
# ---------------------------------------------------------------------------


def _operand_zero_directive(
    a_str: str, op: str, b_str: str,
) -> Tuple[Optional[str], str]:
    """Leading-zero / trailing-zero domain indicator for a pair of operands.

    A leading-zero operand (``"08"``) is only meaningful after reversal; a
    trailing-zero operand (``"50"``) points to the raw domain (reversal
    would have produced ``"05" = 5``, which the author would have written
    as ``"5"``). Train data never mixes the two within one operand pair
    nor uses ``"00"``; assert that invariant.

    Returns ``(directive, base_line)``:

    * ``directive`` -- ``"reversed"`` if any operand has a leading zero,
      ``"normal"`` if any has a trailing zero, ``None`` otherwise.
    * ``base_line`` -- descriptive CoT line without any ``Filter for ...``
      suffix; caller appends that suffix when it also applies the filter.
    """
    has_leading = any(len(s) >= 2 and s[0] == "0" for s in (a_str, b_str))
    has_trailing = any(len(s) >= 2 and s[-1] == "0" for s in (a_str, b_str))
    assert not (has_leading and has_trailing), (
        f"operands {a_str!r},{b_str!r} mix leading and trailing zeros; "
        "training data never contains this combination"
    )
    lead_word = "present" if has_leading else "not present"
    trail_word = "present" if has_trailing else "not present"
    base = (
        f"Let's check if {a_str}{op}{b_str} has zeros. "
        f"So leading zero is {lead_word}, and trailing zero is {trail_word}."
    )
    if has_leading:
        return "reversed", base
    if has_trailing:
        return "normal", base
    return None, base


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LHS_RE = re.compile(r"^\s*(\d+)\s*([^\d\s=])\s*(\d+)\s*$")


def _parse_lhs(lhs: str) -> Optional[Tuple[str, str, str]]:
    m = _LHS_RE.match(lhs)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _parse_output(out_str: str) -> Tuple[int, bool, str, Optional[str], str]:
    """Parse output field.

    Returns ``(value, is_neg, digits, operand_char, mode)`` where ``mode`` is
    ``"plain"`` (positive or plain ``-N``), ``"pre"`` (``opN``), or
    ``"post"`` (``Nop``). ``digits`` is the non-sign digit substring.
    """
    s = out_str.strip()
    if re.fullmatch(r"\d+", s):
        return int(s), False, s, None, "plain"
    if re.fullmatch(r"-\d+", s):
        return int(s), True, s[1:], None, "plain"
    if len(s) >= 2 and not s[0].isdigit() and s[0] != "-" and re.fullmatch(r"\d+", s[1:]):
        digits = s[1:]
        return -int(digits), True, digits, s[0], "pre"
    if len(s) >= 2 and not s[-1].isdigit() and re.fullmatch(r"\d+", s[:-1]):
        digits = s[:-1]
        return -int(digits), True, digits, s[-1], "post"
    return 0, False, "0", None, "plain"


def _parse_prompt(
    prompt: str,
) -> Tuple[List[Tuple[str, str, str, str]], Optional[Tuple[str, str, str]]]:
    """Return ``(examples, query)``.

    ``examples`` is a list of ``(a_str, op, b_str, out_str)`` tuples.
    ``query`` is ``(a_str, op, b_str)`` or None.
    """
    examples: List[Tuple[str, str, str, str]] = []
    query: Optional[Tuple[str, str, str]] = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low.startswith(("in alice", "please put")):
            continue
        m = re.match(r"(?:now,?\s+)?determine the result for:\s*(.+)$",
                     stripped, re.IGNORECASE)
        if m:
            parsed_q = _parse_lhs(m.group(1).strip())
            if parsed_q:
                query = parsed_q
            continue
        if "=" in stripped:
            lhs_part, _, rhs_part = stripped.partition("=")
            parsed_lhs = _parse_lhs(lhs_part)
            if parsed_lhs is None:
                continue
            a, op, b = parsed_lhs
            examples.append((a, op, b, rhs_part.strip()))
    return examples, query


# ---------------------------------------------------------------------------
# Display mode detection (for final answer formatting)
# ---------------------------------------------------------------------------

def _detect_display_mode(
    outputs: List[str],
) -> Tuple[str, Optional[str]]:
    """Scan example outputs for operand markers.

    Returns ``(mode, operand_char)`` where ``mode`` is ``"plain"``,
    ``"prefix"``, or ``"suffix"``. ``operand_char`` is the negative-marker
    character found in examples (None if no examples have operand markers).
    """
    op_char: Optional[str] = None
    mode: str = "plain"
    for out in outputs:
        _, _, _, ch, m = _parse_output(out)
        if m == "pre" and ch:
            mode = "prefix"
            op_char = ch
            return mode, op_char
        if m == "post" and ch:
            mode = "suffix"
            op_char = ch
            return mode, op_char
    return mode, op_char


def _format_final(value: int, is_neg: bool, display_mode: str,
                  op_char: Optional[str]) -> str:
    """Format an answer value using the detected display_mode.

    For positive values, return the plain digit string. For negative values,
    use pre-op (``opN``), post-op (``Nop``), or plain (``-N``) form.
    """
    if not is_neg:
        return str(value)
    abs_str = str(abs(value))
    if display_mode == "prefix" and op_char:
        return op_char + abs_str
    if display_mode == "suffix" and op_char:
        return abs_str + op_char
    return f"-{abs_str}"


def _format_final_digits(digits: str, is_neg: bool, display_mode: str,
                         op_char: Optional[str]) -> str:
    """Like :func:`_format_final` but takes a digit string directly so
    leading zeros from ``R()`` outputs (e.g. ``"0031"``) are preserved."""
    if not is_neg:
        return digits
    if display_mode == "prefix" and op_char:
        return op_char + digits
    if display_mode == "suffix" and op_char:
        return digits + op_char
    return f"-{digits}"


# ---------------------------------------------------------------------------
# Trial body rendering
# ---------------------------------------------------------------------------

def _trial_body(rule_id: str, a_str: str, b_str: str) -> Tuple[str, str]:
    """Return ``(body, sep)`` for the trial line body and its trailing
    separator (``"."`` or ``"?"`` for cat rules)."""
    a = int(a_str)
    b = int(b_str)
    if rule_id == "a-b":
        return f"{a} - {b} = {a - b}", "."
    if rule_id == "|a-b|":
        return f"|{a} - {b}| = {abs(a - b)}", "."
    if rule_id == "b-a":
        diff = a - b
        return f"-({a} - {b}) = -({diff}) = {-diff}", "."
    if rule_id == "-|a-b|":
        d = abs(a - b)
        return f"-|{a} - {b}| = -({d}) = {-d}", "."
    if rule_id == "a+b":
        return f"{a} + {b} = {a + b}", "."
    if rule_id == "a+b+1":
        return f"{a} + {b} = {a + b}, {a + b} + 1 = {a + b + 1}", "."
    if rule_id == "a+b-1":
        return f"{a} + {b} = {a + b}, {a + b} - 1 = {a + b - 1}", "."
    if rule_id == "a*b":
        return f"{a} * {b} = {a * b}", "."
    if rule_id == "a*b+1":
        return f"{a} * {b} = {a * b}, {a * b} + 1 = {a * b + 1}", "."
    if rule_id == "a*b-1":
        return f"{a} * {b} = {a * b}, {a * b} - 1 = {a * b - 1}", "."
    if rule_id == "cat(a,b)":
        L = max(len(a_str), len(b_str), 2)
        res = a_str.zfill(L) + b_str.zfill(L)
        return f"{a_str} and {b_str} maybe combine to {res}", "?"
    if rule_id == "cat(b,a)":
        L = max(len(a_str), len(b_str), 2)
        res = b_str.zfill(L) + a_str.zfill(L)
        return f"{a_str} and {b_str} maybe combine to {res}", "?"
    if rule_id == "max%min":
        if a >= b:
            if b == 0:
                return f"{a} >= {b}. {a} % {b} = undefined", "."
            q = a // b
            prod = b * q
            rem = a - prod
            return (
                f"{a} >= {b}. {b} * {q} = {prod}. "
                f"{a} - {prod} = {rem}",
                ".",
            )
        if a == 0:
            return f"{a} < {b}. {b} % {a} = undefined", "."
        q = b // a
        prod = a * q
        rem = b - prod
        return (
            f"{a} < {b}. {a} * {q} = {prod}. "
            f"{b} - {prod} = {rem}",
            ".",
        )
    return "", "."


def _trial_line(rule_id: str, a_str: str, b_str: str,
                target: Any, matches: bool) -> str:
    body, sep = _trial_body(rule_id, a_str, b_str)
    verdict = f"Yes {target}" if matches else f"Not {target}"
    return f"- {_DISPLAY_NAME[rule_id]}? {body}{sep} {verdict}."


# ---------------------------------------------------------------------------
# Per-example rendering
# ---------------------------------------------------------------------------

def _render_example(
    idx: int, a_str: str, b_str: str, op: str, out_str: str,
    non_concat_normal: List[str],
    non_concat_reversed: List[str],
    concat_surviving: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Render one example block. Returns
    ``(lines, new_normal_other, new_reversed_other, new_concat)``.

    Section order: reversed (non-concat) -> normal (non-concat, prefixed by
    a reversal-back mapping) -> concat (raw trials, no domain). Operand-zero
    filter applies after all sections; it only narrows non-concat lists.
    Concat is domain-free in this catalog (2 ops total) and never dropped
    by the zero filter.
    """
    expected_val, expected_is_neg, expected_digits, _, out_mode = _parse_output(out_str)
    rev_expected_val = _reversed_target_value(expected_digits, expected_is_neg)

    lines: List[str] = [f"Example {idx}"]

    if out_mode in ("pre", "post"):
        lines.append(f"{a_str}{op}{b_str} = {out_str} or {expected_val}")
    else:
        lines.append(f"{a_str}{op}{b_str} = {out_str}")

    L = max(len(a_str), len(b_str), 2)
    sra_str = _sr_str(a_str, L)
    srb_str = _sr_str(b_str, L)

    if expected_is_neg:
        raw_target_disp = f"-{expected_digits}"
        rev_target_disp = f"-{expected_digits[::-1]}"
    else:
        raw_target_disp = expected_digits
        rev_target_disp = expected_digits[::-1]

    # -- Reversed section --
    lines.append("")
    rev_map = _domain_map_phrase("reversed", [
        (a_str, sra_str),
        (b_str, srb_str),
        (raw_target_disp, rev_target_disp),
    ])
    lines.append(f"Maybe the expression is reversed? {rev_map}. Let's test.")

    new_reversed_other: List[str] = []
    if non_concat_reversed:
        for rule_id in non_concat_reversed:
            m = _matches_reversed(
                rule_id, a_str, b_str,
                expected_val, expected_digits, expected_is_neg,
            )
            lines.append(_trial_line(
                rule_id, sra_str, srb_str, rev_expected_val, m,
            ))
            if m:
                new_reversed_other.append(rule_id)
    else:
        lines.append("- not.")
    lines.append(f"So maybe the operator {op} is {'' if new_reversed_other else 'not '}reversed.")

    # -- Normal section --
    lines.append("")
    lines.append("Maybe the expression is normal? Let's examine.")

    new_normal_other: List[str] = []
    if non_concat_normal:
        for rule_id in non_concat_normal:
            m = _matches_normal(
                rule_id, a_str, b_str,
                expected_val, expected_digits, expected_is_neg,
            )
            lines.append(_trial_line(rule_id, a_str, b_str, expected_val, m))
            if m:
                new_normal_other.append(rule_id)
    else:
        lines.append("- not.")
    lines.append(f"So maybe the operator {op} is {'' if new_normal_other else 'not '}normal.")

    # -- Concat section (raw trials, no domain) --
    lines.append("")
    lines.append("Maybe the expression is concatenation? Let's examine.")
    new_concat: List[str] = []
    if concat_surviving:
        # Concat preserves leading zeros, so compare against the raw digit
        # string (or signed string) rather than the int form expected_val.
        concat_target = expected_digits if not expected_is_neg else f"-{expected_digits}"
        for rule_id in concat_surviving:
            m = _matches_normal(
                rule_id, a_str, b_str,
                expected_val, expected_digits, expected_is_neg,
            )
            lines.append(_trial_line(rule_id, a_str, b_str, concat_target, m))
            if m:
                new_concat.append(rule_id)
    else:
        lines.append("- not.")
    lines.append("")

    # Operand-zero filter: prunes non-concat survivors of the losing domain.
    # Concat is domain-free and never dropped here.
    directive, zero_line = _operand_zero_directive(a_str, op, b_str)
    if directive == "reversed":
        lines.append(f"{zero_line} Filter for reversed and concatenation.")
        new_normal_other = []
    elif directive == "normal":
        lines.append(f"{zero_line} Filter for normal and concatenation.")
        new_reversed_other = []
    else:
        lines.append(zero_line)

    # Remaining block
    lines.append("")
    lines.append(f"Now rule for {op} might be")
    if not new_normal_other and not new_reversed_other and not new_concat:
        lines.append("- not.")
    else:
        for rid in new_reversed_other:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in new_normal_other:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in new_concat:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")

    return lines, new_normal_other, new_reversed_other, new_concat


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

_BUCKET_PRIORITY: List[str] = [
    "subtraction",
    "addition",
    "multiplication",
]


def _covered_buckets(rule_ids: List[str]) -> List[str]:
    """Return the single highest-priority bucket covered by ``rule_ids`` as a
    one-element list, or ``[]`` if none of the priority buckets are covered.

    Plan priority is ``_BUCKET_PRIORITY`` ([sub, add, mul]). When an op spans
    multiple priority buckets, only the top-ranked one is reported -- the
    assumption is that the op will ultimately resolve to its highest-priority
    bucket, so that's the only one another symbol needs to yield.
    """
    seen = {_BUCKET[rid] for rid in rule_ids}
    for b in _BUCKET_PRIORITY:
        if b in seen:
            return [b]
    return []


# ---------------------------------------------------------------------------
# Final deduction rendering
# ---------------------------------------------------------------------------

def _apply_chosen(
    rule_id: str, is_reversed: bool, a_str: str, b_str: str,
) -> Tuple[Any, str, bool]:
    """Apply the chosen rule to the query operands.

    Returns ``(inner, output_digits, output_is_neg)``:

    * ``inner``: the rule's raw result — int for numeric rules, str for
      cat rules (the unreversed concatenation).
    * ``output_digits``: the non-sign digit string of the final output
      (preserves leading zeros from the ``R()`` wrapper).
    * ``output_is_neg``: whether the output is negative.
    """
    L = max(len(a_str), len(b_str), 2)
    if is_reversed:
        sra_str = _sr_str(a_str, L)
        srb_str = _sr_str(b_str, L)
        if rule_id in _CAT_RULES:
            inner = _compute_cat(rule_id, sra_str, srb_str)
            return inner, inner[::-1], False
        inner = _compute_numeric(rule_id, int(sra_str), int(srb_str))
        if inner is None:
            return None, "0", False
        # _R_str(inner) preserves leading zeros: "1300" -> "0031".
        if inner < 0:
            return inner, str(-inner)[::-1], True
        return inner, str(inner)[::-1], False
    if rule_id in _CAT_RULES:
        result = _compute_cat(rule_id, a_str, b_str)
        return result, result, False
    result = _compute_numeric(rule_id, int(a_str), int(b_str))
    if result is None:
        return None, "0", False
    if result < 0:
        return result, str(-result), True
    return result, str(result), False


def _final_deduction(
    query_op: str, rule_id: str, is_reversed: bool,
    a_str: str, b_str: str, example_outputs: List[str],
    display_mode: str, op_char: Optional[str],
) -> Tuple[str, str]:
    """Render the final deduction paragraph. Returns (paragraph, predicted)."""
    name = _DISPLAY_NAME[rule_id]
    full_name = f"{name} reversed" if is_reversed else name
    domain_word = "reversed" if is_reversed else "normal"

    L = max(len(a_str), len(b_str), 2)
    if is_reversed:
        body_a = _sr_str(a_str, L)
        body_b = _sr_str(b_str, L)
    else:
        body_a = a_str
        body_b = b_str
    body, sep = _trial_body(rule_id, body_a, body_b)

    inner, output_digits, output_is_neg = _apply_chosen(
        rule_id, is_reversed, a_str, b_str,
    )

    if rule_id in _CAT_RULES:
        # Cat: inner and output are strings; answer is the output string.
        inner_display = str(inner)
        output_display = output_digits
        formatted = output_digits
        predicted = output_digits
    else:
        # Numeric rule
        inner_display = str(inner)
        if output_is_neg:
            output_display = f"-{output_digits}"
        else:
            output_display = output_digits
        formatted = _format_final_digits(
            output_digits, output_is_neg, display_mode, op_char,
        )
        predicted = formatted

    # Reassemble sentence. Prefix with operand-transformation lines (identity
    # in the normal domain, digit-reversal in the reversed domain) so the
    # paragraph opens by stating how the operands map before applying the rule.
    pieces: List[str] = []
    operand_map = _domain_map_phrase(
        domain_word, [(a_str, body_a), (b_str, body_b)],
    )
    pieces.append(
        f"Now compute {a_str}{query_op}{b_str} with {full_name}. "
        f"{operand_map}. "
        f"{body}{sep} "
        f"{inner_display} {domain_word} is {output_display}."
    )

    # "But wait, check example outputs" block
    if example_outputs:
        listing = ", ".join(example_outputs)
        pieces.append(
            f" But wait, maybe we need to check example outputs. "
            f"They are {listing}."
        )
    else:
        pieces.append(" But wait, maybe we need to check example outputs.")

    if op_char is not None and display_mode in ("prefix", "suffix"):
        pos = "front" if display_mode == "prefix" else "back"
        pieces.append(
            f" So if {output_display} is negative, the operator {op_char} "
            f"should be in {pos}."
        )
    else:
        assert False, (
            f"Unreachable: by the time _final_deduction runs, _render_cot has "
            f"already applied the (op_char is None → op_char=q_op, "
            f"display_mode='prefix') fallback, so op_char must be non-None and "
            f"display_mode must be 'prefix' or 'suffix'. "
            f"Got op_char={op_char!r}, display_mode={display_mode!r}, "
            f"query_op={query_op!r}."
        )

    pieces.append(f" Thus {output_display} -> {formatted}")

    return "".join(pieces), predicted


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render_cot(
    examples: List[Tuple[str, str, str, str]],
    query: Tuple[str, str, str],
) -> Tuple[str, Dict[str, Any]]:
    q_a, q_op, q_b = query

    # Display mode detection — needed early for filter 4 (sub-family direction
    # prune) below. The op_char fallback (when no example evidences a negative
    # marker) defaults to the query operator symbol and prefix display, matching
    # the common Kaggle convention; for filter 4 this collapses "plain" into
    # "prefix" (both mean "not suffix"), so the fallback is harmless there.
    example_out_strs = [out for _, _, _, out in examples]
    display_mode, op_char = _detect_display_mode(example_out_strs)
    if op_char is None:
        op_char = q_op
        display_mode = "prefix"

    # Per-operator surviving rule sets. State per op is
    # ``(non_concat_normal, non_concat_reversed, concat)`` -- concat is a
    # flat 2-op set with no domain.
    empty_state = (
        list(_NON_CONCAT_RULE_IDS),
        list(_NON_CONCAT_RULE_IDS),
        list(_CONCAT_RULE_IDS),
    )
    state: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for a_str, op, b_str, _ in examples:
        if op not in state:
            state[op] = (
                list(_NON_CONCAT_RULE_IDS),
                list(_NON_CONCAT_RULE_IDS),
                list(_CONCAT_RULE_IDS),
            )
    if q_op not in state:
        state[q_op] = (
            list(_NON_CONCAT_RULE_IDS),
            list(_NON_CONCAT_RULE_IDS),
            list(_CONCAT_RULE_IDS),
        )

    _parsed_outs = [(o, _parse_output(o)) for _, _, _, o in examples]
    _means_clause = (
        "Possibly operators in the output mean negative sign? So "
        + ", ".join(
            f"{out_str} means {val if mode in ('pre', 'post') else out_str}"
            for out_str, (val, _neg, _dig, _ch, mode) in _parsed_outs
        )
        + ". "
    )

    lines: List[str] = [
        "We need to infer transformation rules from examples. They look "
        "like arithmetic operators but results are weird numbers. Possibly "
        "they mean something like concatenation, subtraction, addition, multiplication? "
        f"{_means_clause}"
        "Let's parse examples.\n"
        "",
    ]

    # Walk examples in prompt order. For each, render the example block
    # against the *current* surviving set for that example's operator, and
    # update the state.
    for idx, (a_str, op, b_str, out_str) in enumerate(examples, start=1):
        cur_n, cur_r, cur_c = state[op]
        block, new_n, new_r, new_c = _render_example(
            idx, a_str, b_str, op, out_str, cur_n, cur_r, cur_c,
        )
        lines.extend(block)
        lines.append("")
        state[op] = (new_n, new_r, new_c)

    # Filter section
    lines.append(
        f"Now we need to apply these rules to the query: {q_a}{q_op}{q_b}\n\n"
        "But wait, "
        f"we don't know the exact operation. We need to apply five rules "
        f"and filter the correct one."
    )
    lines.append("")

    # Build the ordered list of operators to display in the filter section:
    # query op first, then the other operators in the order first seen in
    # examples.
    other_ops: List[str] = []
    for a_str, op, b_str, _ in examples:
        if op != q_op and op not in other_ops:
            other_ops.append(op)
    display_ops = [q_op] + other_ops

    # Per-operator remaining match blocks
    lines.append("First filter")
    lines.append("")
    directives: Dict[str, Optional[str]] = {}
    for op in display_ops:
        n_list, r_list, c_list = state.get(op, empty_state)
        lines.append(f"Now rule for {op} might be")
        if not n_list and not r_list and not c_list:
            lines.append("- not.")
        else:
            for rid in r_list:
                lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
            for rid in n_list:
                lines.append(f"- {_DISPLAY_NAME[rid]}?")
            for rid in c_list:
                lines.append(f"- {_DISPLAY_NAME[rid]}?")
        lines.append("")
        # Domain vote uses non-concat survivors only; concat is silent.
        has_n = bool(n_list)
        has_r = bool(r_list)
        rev_clause = f"the operator {op} is {'' if has_r else 'not '}reversed"
        norm_clause = f"the operator {op} is {'' if has_n else 'not '}normal"
        if has_n and not has_r:
            suffix = " So filter for normal and concatenation."
            directive = "normal"
        elif has_r and not has_n:
            suffix = " So filter for reversed and concatenation."
            directive = "reversed"
        else:
            suffix = ""
            directive = None
        lines.append(f"So maybe {rev_clause}, and maybe {norm_clause}.{suffix}")
        lines.append("")
        directives[op] = directive

    q_normal, q_reversed, q_concat = state[q_op]
    q_op_present = any(op == q_op for _, op, _, _ in examples)

    # Filter 1: domain consensus from non-concat votes (concat doesn't vote).
    op_dirs = [directives[op] for op in [q_op] + other_ops]
    non_none = [d for d in op_dirs if d is not None]
    if len(set(non_none)) > 1:
        assert False, (
            f"inconsistent domain directives {op_dirs} for q_op={q_op!r}"
        )
    filter1_domain = non_none[0] if non_none else None

    if filter1_domain == "normal":
        q_normal_f1 = list(q_normal)
        q_reversed_f1: List[str] = []
    elif filter1_domain == "reversed":
        q_normal_f1: List[str] = []
        q_reversed_f1 = list(q_reversed)
    else:
        q_normal_f1 = list(q_normal)
        q_reversed_f1 = list(q_reversed)
    q_concat_f1 = list(q_concat)

    lines.append(f"Thus the first filtered rules for {q_op} are")
    if not q_normal_f1 and not q_reversed_f1 and not q_concat_f1:
        assert False, "No surviving rules at all! [filter 1]"
    else:
        for rid in q_reversed_f1:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in q_normal_f1:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in q_concat_f1:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
    lines.append("")

    # Filter 2: query-operand-zero. Same rule as the per-example filter,
    # applied directly to (q_a, q_b). Concat is domain-free and unaffected.
    # The reversed-default fires only when filter 1 left both domains
    # populated AND the query has no zero signal.
    lines.append("Second filter")
    lines.append("")
    q_directive, q_zero_line = _operand_zero_directive(q_a, q_op, q_b)
    q_normal_f2, q_reversed_f2, q_concat_f2 = q_normal_f1, q_reversed_f1, q_concat_f1
    if q_directive == "reversed":
        lines.append(f"{q_zero_line} So filter for reversed and concatenation.")
        q_normal_f2 = []
    elif q_directive == "normal":
        lines.append(f"{q_zero_line} So filter for normal and concatenation.")
        q_reversed_f2 = []
    elif q_normal_f2 and q_reversed_f2:
        lines.append(f"{q_zero_line} So filter for reversed and concatenation.")
        q_normal_f2 = []
    else:
        lines.append(f"{q_zero_line} So keep.")
    lines.append("")

    lines.append(f"Thus the second filtered rules for {q_op} are")
    if not q_normal_f2 and not q_reversed_f2 and not q_concat_f2:
        assert False, "No surviving rules at all [filter 2]!"
    else:
        for rid in q_reversed_f2:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in q_normal_f2:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in q_concat_f2:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
    lines.append("")

    # Each non-query op "takes" its top-priority takeable bucket from the
    # query. _covered_buckets reports at most one bucket; concat is not in
    # _BUCKET_PRIORITY so it never claims a bucket. (max%min lives in the
    # subtraction bucket as of the catalog rename, so a mod-only example op
    # now claims subtraction.) Filter 3 drops only non-concat q_op survivors
    # whose bucket was taken. We walk display_ops (q_op first, then
    # other_ops) so the listing mirrors f1's order; q_op contributes no
    # bucket.
    buckets_taken: List[str] = []
    sentences = [f"Check which operator is not used for the query {q_a}{q_op}{q_b}."]
    for op in display_ops:
        if op == q_op:
            sentences.append(f"The operator {op} is the query.")
            continue
        n_list, r_list, c_list = state.get(op, ([], [], []))
        covered = _covered_buckets(n_list + r_list + c_list)
        if covered:
            bucket = covered[0]
            sentences.append(f"The operator {op} is {bucket}, so {bucket} is used.")
            if bucket not in buckets_taken:
                buckets_taken.append(bucket)
        elif c_list:
            # sentences.append(f"The operator {op} is concatenation.")
            pass
    all_categories = _BUCKET_PRIORITY + ["concatenation"]
    available = [b for b in all_categories if b not in buckets_taken]
    if available:
        all_str = ", ".join(all_categories)
        avail_str = ", ".join(available)
        if buckets_taken:
            sentences.append(
                f"So filter out {', '.join(buckets_taken)} from {all_str}."
            )
        else:
            sentences.append(f"So filter from {all_str}.")
        sentences.append(f"So {avail_str} are not used.")
        # sentences.append(f"So keep {avail_str}.")
    else:
        assert False, "No buckets available for query! [filter 3]"
    lines.append("Third filter")
    lines.append("")
    lines.append(" ".join(sentences))
    lines.append("")

    # Filter 3: drop bucket-covered non-concat survivors. Concat passes
    # through untouched.
    q_normal_f3 = [r for r in q_normal_f2 if _BUCKET[r] not in buckets_taken]
    q_reversed_f3 = [r for r in q_reversed_f2 if _BUCKET[r] not in buckets_taken]
    q_concat_f3 = list(q_concat_f2)

    is_guess = not q_normal_f3 and not q_reversed_f3 and not q_concat_f3
    if is_guess:
        import warnings
        q_buckets = _covered_buckets(q_normal_f2 + q_reversed_f2 + q_concat_f2)
        warnings.warn(
            f"equation_transform: q_op {q_op!r} present but every remaining "
            f"bucket {q_buckets} is covered by other ops -- falling back to "
            f"a guess.",
            stacklevel=2,
        )

    lines.append(f"Thus the third filtered rules for {q_op} are")
    if not q_normal_f3 and not q_reversed_f3 and not q_concat_f3:
        lines.append("- not.")
    else:
        for rid in q_reversed_f3:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in q_normal_f3:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in q_concat_f3:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
    lines.append("")

    # Filter 4: subtraction-direction prune from display mode. The dataset
    # encodes subtraction direction via marker position in reversed-domain
    # (prefix `op N` ↔ a-b direction; suffix `N op` ↔ b-a direction), and uses
    # prefix as a fixed convention in normal-domain. So the suffix-aligned and
    # prefix-aligned signed sub rules are mutually exclusive within a problem:
    #
    #   suffix display → drop `a-b`  (the prefix-aligned signed rule)
    #   else          → drop `b-a`  (the suffix-aligned signed rule)
    #
    # Breaks the otherwise-undecidable `b-a | -|a-b|` / `a-b | -|a-b|` plan-
    # order tiebreak when one example uniquely fits both rules.
    f4_drop = "a-b" if display_mode == "suffix" else "b-a"
    q_normal_f4 = [r for r in q_normal_f3 if r != f4_drop]
    q_reversed_f4 = [r for r in q_reversed_f3 if r != f4_drop]
    q_concat_f4 = list(q_concat_f3)

    lines.append("Fourth filter")
    lines.append("")
    _f4_observation = {
        "prefix": "We see operator in front of output",
        "suffix": "We see operator in back of output",
        "plain":  "We see no operator in output",
    }[display_mode]
    lines.append(
        f"If we see no operator in output, we need to filter out "
        f"{_DISPLAY_NAME['b-a']}. "
        f"If we see operator in front of output, we need to filter out "
        f"{_DISPLAY_NAME['b-a']}. "
        f"If we see operator in back of output, we need to filter out "
        f"{_DISPLAY_NAME['a-b']}. "
        f"The example outputs are {', '.join(example_out_strs)}. "
        f"{_f4_observation}. "
        f"So we need to filter out {_DISPLAY_NAME[f4_drop]}."
    )
    lines.append("")
    lines.append(f"Thus the fourth filtered rules for {q_op} are")
    if not q_normal_f4 and not q_reversed_f4 and not q_concat_f4:
        lines.append("- not.")
    else:
        for rid in q_reversed_f4:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in q_normal_f4:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in q_concat_f4:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
    lines.append("")

    # Filter 5: symbol-affinity tiebreak. If the query symbol is one of the
    # standard arithmetic glyphs (+, -, *), restrict the survivor set to the
    # corresponding family bucket (addition/subtraction/multiplication, which
    # includes the canonical rule plus its ±1 / sign variants). Filter is a
    # no-op for any other symbol.
    _wanted_bucket = {"+": "addition", "-": "subtraction", "*": "multiplication"}.get(q_op)
    _choice = _wanted_bucket if _wanted_bucket is not None else "all"
    if _wanted_bucket is not None:
        q_normal_f5 = [r for r in q_normal_f4 if _BUCKET[r] == _wanted_bucket]
        q_reversed_f5 = [r for r in q_reversed_f4 if _BUCKET[r] == _wanted_bucket]
        q_concat_f5: List[str] = []
        # If nothing in the wanted bucket survived, gracefully keep all f4.
        if not q_normal_f5 and not q_reversed_f5:
            q_normal_f5 = list(q_normal_f4)
            q_reversed_f5 = list(q_reversed_f4)
            q_concat_f5 = list(q_concat_f4)
    else:
        q_normal_f5 = list(q_normal_f4)
        q_reversed_f5 = list(q_reversed_f4)
        q_concat_f5 = list(q_concat_f4)

    lines.append("Fifth filter")
    lines.append("")
    lines.append(
        "If we see the query operator is +, we want to keep addition. "
        "If we see the query operator is -, we want to keep subtraction. "
        "If we see the query operator is *, we want to keep multiplication. "
        "If we see the query operator is something else, we want to keep all. "
        f"The query operator is {q_op}. So we want to keep {_choice}."
    )
    lines.append("")
    lines.append(f"Thus the fifth filtered rules for {q_op} are")
    if not q_normal_f5 and not q_reversed_f5 and not q_concat_f5:
        lines.append("- not.")
    else:
        for rid in q_reversed_f5:
            lines.append(f"- {_DISPLAY_NAME[rid]} reversed?")
        for rid in q_normal_f5:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
        for rid in q_concat_f5:
            lines.append(f"- {_DISPLAY_NAME[rid]}?")
    lines.append("")

    # First rule in plan order: non-concat normal, then non-concat reversed,
    # then concat (which has no domain).
    chosen_rule: Optional[str] = None
    chosen_is_rev = False
    if q_normal_f5:
        chosen_rule = q_normal_f5[0]
        chosen_is_rev = False
    elif q_reversed_f5:
        chosen_rule = q_reversed_f5[0]
        chosen_is_rev = True
    elif q_concat_f5:
        chosen_rule = q_concat_f5[0]
        chosen_is_rev = False

    if chosen_rule is None:
        chosen_rule = _RULE_IDS[0]
        is_guess = True

    # Final deduction (display_mode + op_char detected at top of function)
    final_text, predicted = _final_deduction(
        q_op, chosen_rule, chosen_is_rev, q_a, q_b,
        example_out_strs, display_mode, op_char,
    )
    lines.append(final_text)

    cot = "\n".join(lines)

    meta: Dict[str, Any] = {
        "chosen_rule": _DISPLAY_NAME[chosen_rule],
        "chosen_domain": "reversed" if chosen_is_rev else "normal",
        "chosen_bucket": _BUCKET[chosen_rule],
        "is_guess": is_guess,
        "display_mode": display_mode,
        "_predicted_answer": predicted,
    }
    return cot, meta


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Set this env var to a path to capture the prompt + generated CoT whenever the
# deduction falls back to a guess (the L982 / final-fallback warning paths).
# Disabled when unset. Records are appended, newest last.
_GUESS_LOG_ENV = "EQUATION_TRANSFORM_GUESS_LOG"


def _dump_guess(prompt: str, cot_text: str) -> None:
    """Append a guessed (prompt, CoT) pair to the guess log, if configured."""
    path = os.environ.get(_GUESS_LOG_ENV)
    if not path:
        return
    record = (
        "===== GUESS =====\n"
        "----- prompt -----\n"
        f"{prompt.rstrip()}\n"
        "----- cot -----\n"
        f"{cot_text.rstrip()}\n\n"
    )
    try:
        # Single write so concurrent generators interleave at record, not line,
        # granularity. 'a' opens in append mode (O_APPEND), so the seek+write is
        # atomic per call on POSIX for reasonably sized records.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(record)
    except OSError:
        # Logging is best-effort; never let it break generation.
        pass


def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    examples, query = _parse_prompt(prompt)
    answer_stripped = (answer or "").strip()

    q_op = query[1] if query else ""
    by_op_count: Dict[str, int] = {}
    for _, op, _, _ in examples:
        by_op_count[op] = by_op_count.get(op, 0) + 1

    if query is None:
        assert False, "Query should always be present!"

    cot_text, render_meta = _render_cot(examples, query)
    predicted = render_meta.pop("_predicted_answer")
    is_guess = render_meta.get("is_guess", False)

    if is_guess:
        _dump_guess(prompt, cot_text)

    predicted_meta: Optional[str] = None if is_guess else predicted
    correct: Optional[bool] = (
        None if predicted_meta is None else (predicted_meta == answer_stripped)
    )

    meta: Dict[str, Any] = {
        "predicted": predicted_meta,
        "correct": correct,
        "query_op": q_op,
        "query_op_examples": by_op_count.get(q_op, 0),
    }
    meta.update(render_meta)
    return cot_text, meta
