"""Synthetic generator for the equation_transform/symbol category.

Each problem hides one rule per operator drawn from a 3-way mix:
``_P_FWD`` concat (``ABCD``, "drop the middle symbol"), ``_P_REV`` reverse
concat (``CDAB``, "swap and drop the middle symbol"), and the remainder
arithmetic (``a+b`` / ``|a-b|`` / ``a*b``). Arithmetic outputs are
length-variable (1-4 chars) and don't match either structural concat, so
the CoT classifies them as ``unknown`` and routes them through the guess
branch -- this is what gets the model to learn the unknown path.

Target case mix on the generated set. The ``none`` rate (query operator
absent from the demos -> rule uninferable) is pinned directly via
:data:`_NONE_RATE` rather than emerging from pool sampling, so each class
share is ``(1 - _NONE_RATE) × per-op-rule-prob``:

    concat              ~25%   (0.85 × _P_FWD)
    concat reversed     ~40%   (0.85 × _P_REV)
    unknown             ~20%   (0.85 × _P_ARITH)
    none                ~15%   (= _NONE_RATE)

Operator pool reuses the numeric generator's standard/non-standard split
(:data:`STANDARD_OPS` ∪ :data:`NONSTANDARD_OPS`, weighted by :data:`P_STD`).

Digits 0-9 in the rendered problem are remapped to characters drawn from the
27-char :data:`SPECIAL` alphabet, excluding any punctuation already present
in the prompt or answer so the targets never collide with op chars or
preamble punctuation (e.g. ``'`` in "Alice's", ``:`` in the query prefix).
"""

import random
from typing import Dict, List, Optional, Tuple

from nemotron.data.main._syn_base import SyntheticGenerator, make_id
from nemotron.data.main.equation_transform_numeric_syn import (
    NONSTANDARD_OPS,
    P_STD,
    STANDARD_OPS,
    _HEADER,
    _sample_op_pool,
    _zfill2,
)

SPECIAL: List[str] = sorted(
    "!\"#$%&'()*+-/:;<>?@[\\]^`{|}", key=ord
)
assert len(SPECIAL) == 27, f"SPECIAL must have 27 chars, got {len(SPECIAL)}"

# Tokenizer-difficulty proxy: number of the 27 SPECIAL partners each glyph
# *fully merges with* into a single token under the Nemotron tokenizer (max
# 47; measured 2026-05-16). Copy-corruption
# (EX_COMPUTE/Q_COMPUTE) tracks this near-linearly — merging is an
# alphabet-wide gradient, not a fixed "hard set", so the per-problem 10-glyph
# alphabet is drawn weighted by it (replaces the old binary _HARD_GLYPHS).
_MERGE_COUNT: Dict[str, int] = {
    ")": 47, "(": 43, '"': 40, "'": 40, "\\": 39, "/": 38, ":": 35, "-": 34,
    "{": 33, "}": 33, "]": 32, "*": 30, ">": 28, "[": 27, "$": 24, "+": 24,
    "<": 23, "%": 21, "`": 20, "!": 19, ";": 19, "?": 17, "^": 15, "#": 13,
    "&": 10, "|": 10, "@": 8,
}
# Alphabet-draw bias toward hard-to-tokenize glyphs. Per-glyph weight is
#   clamp(merge_count, *_MERGE_CLAMP) ** _DIFF_EXP
# The clamp flattens the extreme tiers — the hardest glyphs share one weight
# (broad, even coverage instead of ')' domination) and the easiest aren't
# fully starved; _DIFF_EXP then shapes the gradient between the clamp bounds.
# Both knobs are tunable. Calibrated so the hardest tier appears ~5x more
# often than the easiest in the drawn alphabet (measured empirically):
# (15, 35) + exp 2.5 ≈ 5.1x. _DIFF_EXP=0 →
# uniform; raise exp or widen the clamp for a harder stress set.
_MERGE_CLAMP: Tuple[int, int] = (15, 35)
_DIFF_EXP: float = 2.5  # exp 2.0 ≈ 3.6x, 2.5 ≈ 5.1x, 3.0 ≈ 7.2x


_RULE_FWD = "fwd"     # ABCD: drop the middle symbol
_RULE_REV = "rev"     # CDAB: swap and drop the middle symbol
_RULE_ARITH = "arith"  # a+b / |a-b| / a*b -- routes to ``unknown`` in CoT

_ARITH_SUBS: Tuple[str, ...] = ("add", "sub", "mul")

# Per-op rule split. With the pinned ``none`` rate below, class shares are
# (1 - _NONE_RATE) × these → concat ~25%, concat_rev ~40%, unknown ~20%.
_P_FWD: float = 0.29
_P_REV: float = 0.47
# remainder (~0.24) -> arithmetic

# Knob B: exact P(query operator absent from every demo). When absent the
# reference CoT cannot infer the rule and falls back to a guess (the "none"
# bucket). Pinned here — instead of emerging from pool size / n_demos — so the
# class mix is directly controllable and pool-size-independent.
_NONE_RATE: float = 0.15


# A rule is a (kind, arith_sub) pair. ``arith_sub`` is ``None`` for
# structural rules and one of ``_ARITH_SUBS`` for ``_RULE_ARITH``.
Rule = Tuple[str, Optional[str]]


def _sample_op_rule(rng: random.Random) -> Rule:
    r = rng.random()
    if r < _P_FWD:
        return (_RULE_FWD, None)
    if r < _P_FWD + _P_REV:
        return (_RULE_REV, None)
    return (_RULE_ARITH, rng.choice(_ARITH_SUBS))


def _apply_rule(a_pad: str, b_pad: str, rule: Rule) -> str:
    kind, sub = rule
    if kind == _RULE_FWD:
        return a_pad + b_pad
    if kind == _RULE_REV:
        return b_pad + a_pad
    # arithmetic
    a, b = int(a_pad), int(b_pad)
    if sub == "add":
        return str(a + b)
    if sub == "sub":
        return str(abs(a - b))
    if sub == "mul":
        return str(a * b)
    raise ValueError(f"unknown arith sub-rule: {sub!r}")


def _sample_alphabet(
    rng: random.Random, pool: List[str], k: int = 10,
) -> List[str]:
    """Weighted sample of *k* distinct glyphs, no replacement.

    Each glyph's weight is ``clamp(_MERGE_COUNT[c], *_MERGE_CLAMP) **
    _DIFF_EXP`` — biased toward glyphs the tokenizer most often merges with a
    neighbour (the copy-corruption driver), the clamp flattening the extreme
    tiers. Efraimidis–Spirakis A-Res keying
    (``key = u ** (1/w)``, take the top-k keys) — exact weighted
    sampling-without-replacement in one pass; every glyph stays reachable so
    the eval alphabet isn't truncated. ``_DIFF_EXP=0`` recovers the old
    uniform ``rng.sample``.
    """
    def w(c: str) -> float:
        lo, hi = _MERGE_CLAMP
        return float(max(lo, min(hi, _MERGE_COUNT.get(c, lo)))) ** _DIFF_EXP
    keyed = sorted(
        ((rng.random() ** (1.0 / w(c)), c) for c in pool), reverse=True)
    return [c for _, c in keyed[:k]]


class EquationTransformSymbolGenerator(SyntheticGenerator):
    """Generate symbol problems with mixed concat/rev/arith rules per operator."""

    def __init__(
        self,
        seed: int = 42,
        rule_mix: Optional[str] = None,
    ) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)

    def generate_one(self) -> Dict[str, str]:
        rng = self._rng
        n_demos = rng.randint(3, 5)

        ops = _sample_op_pool(rng)
        op_rule: Dict[str, Rule] = {op: _sample_op_rule(rng) for op in ops}

        # Query operator is uniform over the pool; whether it ALSO appears in
        # the demos is controlled directly (Knob B) so the "none" rate is
        # exactly _NONE_RATE, independent of pool size / n_demos. Absence is
        # impossible with a single-op pool, so presence is forced there.
        qop = rng.choice(ops)
        keep_absent = len(ops) > 1 and rng.random() < _NONE_RATE

        demo_ops: List[str] = []
        for _ in range(n_demos):
            op = rng.choice(ops)
            if keep_absent:
                while op == qop:
                    op = rng.choice(ops)
            demo_ops.append(op)
        if not keep_absent and qop not in demo_ops:
            demo_ops[rng.randrange(n_demos)] = qop  # guarantee inferability

        lines: List[Tuple[str, str, str, str]] = []
        for op in demo_ops:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            a_pad, b_pad = _zfill2(a), _zfill2(b)
            lines.append(
                (a_pad, op, b_pad, _apply_rule(a_pad, b_pad, op_rule[op])))
        qa_i, qb_i = rng.randint(10, 99), rng.randint(10, 99)
        qa_p, qb_p = _zfill2(qa_i), _zfill2(qb_i)
        lines.append(
            (qa_p, qop, qb_p, _apply_rule(qa_p, qb_p, op_rule[qop])))

        examples = "\n".join(f"{a}{op}{b} = {out}" for a, op, b, out in lines[:-1])
        qa, qop, qb, q_out = lines[-1]
        prompt = (
            f"{_HEADER}\n"
            f"{examples}\n"
            f"Now, determine the result for: {qa}{qop}{qb}"
        )

        forbidden = {
            c for c in prompt + q_out
            if not (c.isalnum() or c.isspace() or c == "=")
        }
        pool = [c for c in SPECIAL if c not in forbidden]
        targets = _sample_alphabet(self._rng, pool, 10)
        digit_map = str.maketrans("0123456789", "".join(targets))

        return {
            "id": make_id(),
            "prompt": prompt.translate(digit_map),
            "answer": q_out.translate(digit_map),
        }
