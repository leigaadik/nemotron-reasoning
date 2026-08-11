"""Synthetic generator for the equation_transform/numeric category.

See `nemotron/data/docs/equation_transform_numeric_gen.md`.
"""

import random
from typing import Dict, List, Optional, Tuple

from nemotron.data.main._syn_base import SyntheticGenerator, make_id


STANDARD_OPS: List[str] = ["+", "-", "*"]
NONSTANDARD_OPS: List[str] = [
    '"', "`", "[", "$", "{", "!", "<", "%", "'", "^",
    "\\", "?", "}", "]", "#", ":", "@", ">", "&", "(", ")", "|", "/",
]
assert len(NONSTANDARD_OPS) == 23
BRACKETS: Tuple[str, ...] = ("(", "[", "{")

P_STD = 0.45
P_DOMAIN_RAW = 1.0 / 3.5
# Per-problem (not per-operator) display mode. Every signed example in a
# problem shares one mode, so problem-level and example-level prefix:suffix
# rates coincide.
P_DISPLAY_PRE = 0.50
# Probability that a non-focused problem is required to contain at least one
# signed-output example, so the per-problem mode is actually displayed and
# the CoT's f4 prefix-vs-suffix conclusion fires. The complement is left as
# "plain"-display problems (no signed example; CoT defaults to prefix).
P_FORCE_SIGNED = 0.70
# Probability that a non-concat sR-domain example is rejection-sampled to
# yield a leading-zero output (i.e. result value ≡ 0 (mod 10), whose reversed
# digit string starts with "0"). Exercises the f2 leading-zero branch.
P_LEAD0 = 0.20
P_CAT = 0.065
P_REV_CAT = 0.065
# Equal share for add/sub/mul of the non-concat mass. `max%min` no longer has
# its own bucket — it lives inside the sub bucket (matching the CoT catalog),
# with within-sub probability P_MOD_WITHIN_SUB.
P_ARITH = (1.0 - P_CAT - P_REV_CAT) / 3.0
P_SUB = P_ARITH
P_ADD = P_ARITH
P_MUL = P_ARITH
P_MOD_WITHIN_SUB = 0.10 / P_SUB  # ≈ 0.345 — preserves original P_MOD ≈ 0.10 whole-problem mass

BUCKETS: List[Tuple[str, float]] = [
    ("cat",         P_CAT),
    ("reverse_cat", P_REV_CAT),
    ("add",         P_ADD),
    ("sub",         P_SUB),
    ("mul",         P_MUL),
]

NEG_RULES: Tuple[str, ...] = ("a-b", "b-a", "-|a-b|")
# Kaggle convention: each NEG_RULE co-occurs with exactly one display mode.
# Violating this pair-up makes the CoT's f4 prune the correct rule and fall
# back to is_guess. Keep mode and rule in lockstep.
NEG_RULES_PRE:  Tuple[str, ...] = ("a-b",)
NEG_RULES_POST: Tuple[str, ...] = ("b-a", "-|a-b|")
_CAT_RULES: Tuple[str, ...] = ("cat(a,b)", "cat(b,a)")


def _zfill2(n: int) -> str:
    return f"{n:02d}"


def _sR_int(n: int) -> int:
    return int(_zfill2(n)[::-1])


def _sample_op_pool(rng: random.Random) -> List[str]:
    pool: List[str] = []
    while len(pool) < 3:
        c = rng.choice(STANDARD_OPS) if rng.random() < P_STD else rng.choice(NONSTANDARD_OPS)
        if c not in pool:
            pool.append(c)
    return pool


def _sample_buckets(rng: random.Random) -> List[str]:
    items = [b for b, _ in BUCKETS]
    weights = [w for _, w in BUCKETS]
    out: List[str] = []
    for _ in range(3):
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r < acc:
                out.append(items.pop(i))
                weights.pop(i)
                break
    return out


def _sample_rule(rng: random.Random, bucket: str, mode: Optional[str] = None) -> str:
    """Sample a rule from *bucket*. For the sub bucket, *mode* gates which
    direction-bearing rules are eligible to preserve the Kaggle convention:
    pre allows ``a-b``; post allows ``b-a`` and ``-|a-b|``. Mode-agnostic sub
    rules (``|a-b|``, ``max%min``) are eligible in either mode.
    """
    if bucket == "cat":         return "cat(a,b)"
    if bucket == "reverse_cat": return "cat(b,a)"
    r = rng.random()
    if bucket == "add":
        return "a+b" if r < 0.5 else ("a+b+1" if r < 0.75 else "a+b-1")
    if bucket == "mul":
        return "a*b" if r < 0.5 else ("a*b+1" if r < 0.75 else "a*b-1")
    # sub bucket: max%min absorbs the original P_MOD share, rest splits the
    # original sub-family distribution among mode-eligible directional rules.
    if r < P_MOD_WITHIN_SUB:
        return "max%min"
    r2 = rng.random()
    if mode == "pre":
        # 0.5 a-b, 0.5 |a-b|
        return "a-b" if r2 < 0.5 else "|a-b|"
    if mode == "post":
        # 0.5 b-a, 0.25 |a-b|, 0.25 -|a-b| -- keeps |a-b| share roughly even
        # across modes; -|a-b| inherits its old 1/6 share alongside b-a.
        if r2 < 0.5:        return "b-a"
        if r2 < 0.75:       return "|a-b|"
        return "-|a-b|"
    # mode unknown -- preserve the legacy distribution
    if r2 < 0.5:        return "a-b"
    if r2 < 0.5 + 1/6:  return "b-a"
    if r2 < 0.5 + 2/6:  return "|a-b|"
    return "-|a-b|"


def _sample_operand(rng: random.Random, domain: str) -> int:
    if domain == "raw":
        return rng.randint(10, 99)
    while True:
        v = rng.randint(1, 99)
        if v % 10 != 0:
            return v


def _compute(rule_id: str, domain: Optional[str], a: int, b: int) -> Optional[str]:
    a_pad, b_pad = _zfill2(a), _zfill2(b)
    if rule_id == "cat(a,b)": return a_pad + b_pad
    if rule_id == "cat(b,a)": return b_pad + a_pad

    x, y = (a, b) if domain == "raw" else (_sR_int(a), _sR_int(b))
    if   rule_id == "a-b":      v = x - y
    elif rule_id == "b-a":      v = y - x
    elif rule_id == "|a-b|":    v = abs(x - y)
    elif rule_id == "-|a-b|":   v = -abs(x - y)
    elif rule_id == "a+b":      v = x + y
    elif rule_id == "a+b+1":    v = x + y + 1
    elif rule_id == "a+b-1":    v = x + y - 1
    elif rule_id == "a*b":      v = x * y
    elif rule_id == "a*b+1":    v = x * y + 1
    elif rule_id == "a*b-1":    v = x * y - 1
    elif rule_id == "max%min":
        m = min(x, y)
        if m == 0:
            return None
        v = max(x, y) % m
    else:
        raise ValueError(rule_id)

    mag = -v if v < 0 else v
    digits = str(mag)[::-1] if domain == "sR" else str(mag)
    return ("-" + digits) if v < 0 else digits


def _render(raw_out: str, op: str, mode: str) -> str:
    if not raw_out.startswith("-"):
        return raw_out
    digits = raw_out[1:]
    return op + digits if mode == "pre" else digits + op


def _build_focused_trio(
    rng: random.Random, problem_domain: str, problem_mode: str,
) -> Tuple[List[str], Dict[str, Dict[str, Optional[str]]], str]:
    bracket_op = rng.choice(BRACKETS)
    pool: List[str] = [bracket_op]
    while len(pool) < 3:
        c = rng.choice(STANDARD_OPS) if rng.random() < P_STD else rng.choice(NONSTANDARD_OPS)
        if c not in pool:
            pool.append(c)

    other_pool = [(b, w) for b, w in BUCKETS if b != "sub"]
    items = [b for b, _ in other_pool]
    weights = [w for _, w in other_pool]
    other_buckets: List[str] = []
    for _ in range(2):
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r < acc:
                other_buckets.append(items.pop(i))
                weights.pop(i)
                break

    op_meta: Dict[str, Dict[str, Optional[str]]] = {}
    other_iter = iter(other_buckets)
    for op in pool:
        if op == bracket_op:
            op_meta[op] = {"rule_id": "-|a-b|", "domain": problem_domain, "mode": problem_mode}
        else:
            bucket = next(other_iter)
            domain: Optional[str] = None if bucket in ("cat", "reverse_cat") else problem_domain
            op_meta[op] = {
                "rule_id": _sample_rule(rng, bucket, mode=problem_mode),
                "domain": domain,
                "mode": problem_mode,
            }
    return pool, op_meta, bracket_op


_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:"
)


class EquationTransformNumericGenerator(SyntheticGenerator):

    def __init__(
        self,
        seed: int = 42,
        rule_mix: Optional[str] = None,
        neg_question_weight: float = 10.0,
        bracket_focus_p: float = 0.10,
    ) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)
        self._neg_q_w = float(neg_question_weight)
        self._bracket_focus_p = float(bracket_focus_p)

    def generate_one(self) -> Dict[str, str]:
        rng = self._rng
        n_demos = rng.randint(3, 5)
        problem_domain = "raw" if rng.random() < P_DOMAIN_RAW else "sR"
        problem_mode = "pre" if rng.random() < P_DISPLAY_PRE else "post"

        focused = rng.random() < self._bracket_focus_p
        if focused:
            # Bracket showcase uses -|a-b|, which is post-display by Kaggle
            # convention. Override problem_mode for focused problems so the
            # whole trio stays self-consistent.
            problem_mode = "post"
            ops, op_meta, bracket_op = _build_focused_trio(rng, problem_domain, problem_mode)
            showcase_idx: Optional[int] = rng.randrange(n_demos) if n_demos > 0 else None
        else:
            ops = _sample_op_pool(rng)
            buckets = _sample_buckets(rng)
            op_meta = {}
            for op, bucket in zip(ops, buckets):
                domain: Optional[str] = None if bucket in ("cat", "reverse_cat") else problem_domain
                op_meta[op] = {
                    "rule_id": _sample_rule(rng, bucket, mode=problem_mode),
                    "domain": domain,
                    "mode": problem_mode,
                }
            bracket_op = None
            showcase_idx = None

        # Force-signed example pin (non-focused problems only; focused already
        # have a guaranteed signed bracket-op showcase). If no op currently has
        # a NEG-capable rule, convert one op's rule to a random NEG_RULE so the
        # pin can fire.
        force_signed = (not focused) and rng.random() < P_FORCE_SIGNED
        forced_slot: Optional[int] = None
        forced_op: Optional[str] = None
        if force_signed:
            # Only NEG_RULES matching the problem's display mode count -- a
            # mode-mismatched neg rule would violate the Kaggle convention and
            # confuse f4. (a-b is pre-only; b-a / -|a-b| are post-only.)
            mode_neg_rules = NEG_RULES_PRE if problem_mode == "pre" else NEG_RULES_POST
            neg_ops = [op for op in ops if op_meta[op]["rule_id"] in mode_neg_rules]
            if not neg_ops:
                # Prefer converting an existing non-NEG sub-bucket op (|a-b| or
                # max%min) so the overall bucket distribution stays put;
                # otherwise convert any op.
                sub_non_neg = {"|a-b|", "max%min"}
                candidates = [op for op in ops if op_meta[op]["rule_id"] in sub_non_neg]
                target = rng.choice(candidates) if candidates else rng.choice(ops)
                op_meta[target] = {
                    "rule_id": rng.choice(mode_neg_rules),
                    "domain": problem_domain,
                    "mode": problem_mode,
                }
                neg_ops = [target]
            forced_op = rng.choice(neg_ops)
            forced_slot = rng.randrange(n_demos)

        bias_q = (not focused) and self._neg_q_w != 1.0
        q_weights = (
            [self._neg_q_w if op_meta[op]["rule_id"] in NEG_RULES else 1.0 for op in ops]
            if bias_q else None
        )

        lines: List[Tuple[str, str, str, str]] = []
        for i in range(n_demos + 1):
            if focused and (i == n_demos or i == showcase_idx):
                op = bracket_op
            elif force_signed and i == forced_slot:
                op = forced_op
            elif bias_q and i == n_demos:
                op = rng.choices(ops, weights=q_weights, k=1)[0]
            else:
                op = rng.choice(ops)
            m = op_meta[op]
            require_neg = (focused and op == bracket_op) or (
                force_signed and i == forced_slot
            )
            # Leading-0 only achievable for non-concat sR-domain rules (reversed
            # digit string starts with "0" iff the result value is a multiple
            # of 10). Skip the rejection on rules that can't satisfy it. Soft
            # constraint: try with lead0 first, fall back if exhausted.
            want_lead0 = (
                rng.random() < P_LEAD0
                and m["rule_id"] not in _CAT_RULES
                and m["domain"] == "sR"
            )
            sampled: Optional[Tuple[int, int, str]] = None
            for attempt_want_lead0 in (True, False) if want_lead0 else (False,):
                for _ in range(64):
                    a = _sample_operand(rng, problem_domain)
                    b = _sample_operand(rng, problem_domain)
                    raw = _compute(m["rule_id"], m["domain"], a, b)
                    if raw is None:
                        continue
                    if require_neg and not raw.startswith("-"):
                        continue
                    if attempt_want_lead0:
                        digs = raw[1:] if raw.startswith("-") else raw
                        if not (len(digs) > 1 and digs[0] == "0"):
                            continue
                    sampled = (a, b, raw)
                    break
                if sampled is not None:
                    break
            if sampled is None:
                raise RuntimeError("operand sampling failed")
            a, b, raw = sampled
            lines.append((_zfill2(a), op, _zfill2(b), _render(raw, op, m["mode"])))

        examples = "\n".join(f"{a}{op}{b} = {out}" for a, op, b, out in lines[:-1])
        qa, qop, qb, q_out = lines[-1]
        prompt = (
            f"{_HEADER}\n"
            f"{examples}\n"
            f"Now, determine the result for: {qa}{qop}{qb}"
        )
        return {"id": make_id(), "prompt": prompt, "answer": q_out}
