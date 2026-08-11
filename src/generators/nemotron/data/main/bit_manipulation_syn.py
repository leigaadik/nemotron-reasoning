"""Synthetic generator for the bit_manipulation category.

Rule-picker: draws rules from the tracked catalog
``csvs/bit_manip_rules.json`` (built by
``scripts/discovery/bit_manip_rules_final.py``). Rules are bucketed by family
— ``no_op``, ``one_op``, ``two_op``, ``majority``, ``choice`` — and drawn
uniformly within a family; ``family_mix`` (or the default
``_DEFAULT_FAMILY_MIX``) steers the family-level sampling. ``family_base_mix``
instead enumerates each family's full grammar procedurally (see ``__init__``).
See ``nemotron/data/docs/bit_manipulation.md``. Per call we

  1. Pick a family by the active mix.
  2. Draw a rule from that family.
  3. Pick the example-pair count uniformly from ``{7, 8, 9, 10}``.
  4. Sample ``n_examples + 1`` distinct 8-bit inputs without replacement and
     compute outputs from the rule's 256-entry LUT.
  5. Format the prompt with the template ``bit_manipulation_cot`` parses
     (pair regex ``([01]{8})\\s*->\\s*([01]{8})``, query regex
     ``determine the output for:\\s*([01]{8})``; the header contains the
     substring ``bit manipulation rule`` the router keys on).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemotron.data.main._syn_base import SyntheticGenerator, make_id
from nemotron.data.main.bit_manipulation_cot import _PAIR_RE, classify_fallback


# ---------------------------------------------------------------------------
# Rule parsing — every rule string compiles to a 256-entry uint8 LUT.
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(r"^(ROT|SHR|SHL)([0-7])$")
_ATOM = r"(?:ROT|SHR|SHL)[0-7]"
_OP = r"(?:ANDNOT|ORNOT|XORNOT|AND|OR|XOR)"
_TWO_OP_RE = re.compile(
    rf"^\(\s*({_ATOM})\s+({_OP})\s+({_ATOM})\s*\)\s+({_OP})\s+({_ATOM})$")
_ONE_OP_RE = re.compile(rf"^({_ATOM})\s+({_OP})\s+({_ATOM})$")
_MAJ_RE = re.compile(rf"^Maj\(\s*({_ATOM}),\s*({_ATOM}),\s*({_ATOM})\s*\)$")
_CH_RE = re.compile(rf"^Ch\(\s*({_ATOM}),\s*({_ATOM}),\s*({_ATOM})\s*\)$")

_MASK = 0xFF
_XS = np.arange(256, dtype=np.uint16)

# Default family weights when neither family_mix nor family_base_mix is given
# (the catalog mix; mirrors final.yaml's documented (A) CATALOG default). Held
# as relative weights — ``random.choices`` normalizes — so they need not sum to 1.
_DEFAULT_FAMILY_MIX: Dict[str, float] = {
    "no_op": 0.08,
    "one_op": 0.16,
    "two_op": 0.24,
    "majority": 0.24,
    "choice": 0.18,
}


def _atom_lut(name: str) -> np.ndarray:
    """ROTn = rotate-left by n; SHRn = x >> n; SHLn = (x << n) & 0xFF."""
    m = _ATOM_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"unrecognized atom: {name!r}")
    fam, n = m.group(1), int(m.group(2))
    if fam == "ROT":
        if n == 0:
            return _XS.astype(np.uint8)
        return (((_XS << n) | (_XS >> (8 - n))) & _MASK).astype(np.uint8)
    if fam == "SHR":
        return (_XS >> n).astype(np.uint8)
    return ((_XS << n) & _MASK).astype(np.uint8)


def _apply(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    if op == "AND":     return a & b
    if op == "ANDNOT": return a & ((~b) & _MASK)
    if op == "OR":      return a | b
    if op == "ORNOT":  return a | ((~b) & _MASK)
    if op == "XOR":     return a ^ b
    if op == "XORNOT": return a ^ ((~b) & _MASK)
    raise ValueError(f"unrecognized op: {op!r}")


def rule_to_lut(rule: str) -> np.ndarray:
    """Compile a rule string to its full 256-entry output LUT.

    Order matters: the bracketed/`Maj(`/`Ch(` forms are tested before the
    bare ``A OP B`` (one_op) form so a maj/ch string is never mis-parsed.
    """
    m = _TWO_OP_RE.match(rule)
    if m:
        p1, o1, p2, o2, p3 = m.groups()
        return _apply(_apply(_atom_lut(p1), _atom_lut(p2), o1),
                      _atom_lut(p3), o2)
    m = _MAJ_RE.match(rule)
    if m:
        a, b, c = (_atom_lut(s) for s in m.groups())
        return (a & b) | (b & c) | (a & c)
    m = _CH_RE.match(rule)
    if m:
        a, b, c = (_atom_lut(s) for s in m.groups())
        return (a & b) | (((~a) & _MASK) & c)
    m = _ONE_OP_RE.match(rule)
    if m:
        p1, o1, p2 = m.groups()
        return _apply(_atom_lut(p1), _atom_lut(p2), o1)
    if _ATOM_RE.fullmatch(rule):
        return _atom_lut(rule)
    raise ValueError(f"unparseable rule: {rule!r}")


def _gte_lut(a: int, b: int, c: int) -> np.ndarray:
    """LUT for the Shape-C "unknown" rule: ``(SHLa XORNOT ROTb) OR SHRc`` with its
    all-ones cell cleared — ``ANDNOT (SHLa AND ROTb AND SHRc)``. Per bit this is
    ``g(L,T,R) = XNOR(L,T) if SHR=0 else NAND(L,T)``. It matches Shape A on every
    corner and differs only on the live-middle (1,1,1) cells, so it is outside the
    signature grammar (which has Shape A/B but not this combiner)."""
    shape_a = rule_to_lut(f"(SHL{a} XORNOT ROT{b}) OR SHR{c}")
    all_ones = _atom_lut(f"SHL{a}") & _atom_lut(f"ROT{b}") & _atom_lut(f"SHR{c}")
    return shape_a & ((~all_ones) & _MASK)


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
    "\n\n"
    "Here are some examples of input -> output:\n"
    "{pairs}"
    "\n\n"
    "Now, determine the output for: {query}"
)

# Legacy kwargs from the old CSV-replayer signature; accepted and ignored so
# pre-existing configs (e.g. gen_kwargs.bit_manipulation.hard_ids_path) do not
# raise TypeError against the new rule-picker signature.
_LEGACY_KWARGS = ("hard_ids_path", "hard_multiplicity", "train_csv_path")


def _default_rules_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "csvs", "bit_manip_rules.json")
    )


# Maps user-facing family_mix keys to the family strings used in
# csvs/bit_manip_rules.json (entry["family"][0]).
_FAMILY_KEY_TO_JSON: Dict[str, str] = {
    "no_op":    "no-op",
    "one_op":   "one-op",
    "two_op":   "2-op",
    "majority": "maj",
    "choice":   "ch",
}

# Pseudo-family for the "unknown" rows the signature solver gives up on (the
# ~3.3% of train.csv with no derivable rule). They are NOT noised and NOT
# structureless: each is a clean Shape-C live-middle rule
# ``(SHLa XORNOT ROTb) OR SHRc`` with its all-ones cell cleared
# (``g(L,T,R) = XNOR(L,T) if SHR=0 else NAND(L,T)``, a+c<=7), a third 2-op
# combiner the signature grammar lacks. It agrees with Shape A on the corners
# the solver reads, so corner detection succeeds but the middle never matches by
# exact grammar — it falls through to the fallback (which recovers it as GTE).
# Generated on the fly (these have no catalog rule); selected via family_mix.
_UNKNOWN_KEY = "unknown"
# Vestigial: earlier the unknown family was modelled as noised 2-op (this was the
# per-middle-bit flip probability). Shape C is exact, so this is unused; kept only
# as the default for the accepted-but-ignored ``noise_p`` constructor arg.
_DEFAULT_NOISE_P = 0.09

# `overrun_p` forcing (see __init__/generate_one). Live-middle families whose
# corner runs can over-run a region boundary — the only families that can land in
# the solver's block-1 "spurious over-run" fallback. no_op and one_op
# (trivial/full-width corners) and `unknown` (Shape-C -> block-2) are excluded.
_LIVE_MIDDLE_CATS = frozenset({"two_op", "majority", "choice"})
# Cap on biased redraws per forced slot before failing loud (a guarantee must not
# silently emit a non-over-run row). Block-1 is an intrinsically rare COINCIDENCE
# in the sampled example columns (~0.3% even on the live-middle + n_ex=7 proposal —
# no rule family reliably forces it), so a forced slot needs a few hundred draws on
# average; the cap must clear the geometric tail (P(miss)=(1-0.003)^5000 ~ 3e-7 per
# slot) so a real run never spuriously raises. The mean cost, not the cap, sets the
# wall-clock (~106s extra on a 7200-row run at overrun_p=0.05). If this proves too
# slow, pre-mine a block-1 instance pool offline and draw forced slots from it.
_OVERRUN_MAX_TRIES = 5000

# csvs/bit_manip_rules.json stores 2-op rules with the tilde-NOT shorthand
# (e.g. "(SHR1 OR ~ROT7) XOR SHL1"); rule_to_lut's grammar uses the explicit
# "OPNOT" form. This regex rewrites "OP ~ATOM" -> "OPNOT ATOM" before parse.
_TILDE_NOT_RE = re.compile(r"\b(AND|OR|XOR)\s+~(ROT|SHR|SHL)")


def _normalize_rule_string(rule: str) -> str:
    return _TILDE_NOT_RE.sub(r"\1NOT \2", rule)


# ---------------------------------------------------------------------------
# Procedural grammar enumeration (family_base_mix mode)
# ---------------------------------------------------------------------------
# The full reachable rule universe per BASE family, shifts a,b,c in {1..7}
# (ROT0 excluded, per the docs). Used by procedural mode, which samples a base
# family and then a rule UNIFORMLY from its grammar and emits the problem AS-IS
# — so when a particular (a,b,c) makes the function collapse to a simpler family
# (a=b kills the Left primitive, b+c=8 the Right, a+c=8 empties the middle, dead
# corners, etc.), the emitted problem genuinely IS that simpler function. This
# "degeneracy drift" is intrinsic to the truth table and is what makes the
# resolved-family mix differ from the base-sampling mix (see final.yaml).
_SHIFTS = range(1, 8)
_OPS6 = ("AND", "OR", "XOR", "ANDNOT", "ORNOT", "XORNOT")


def _enumerate_family_rules(key: str) -> List[str]:
    """All grammar rule strings for one base family key (a subset of
    ``_FAMILY_KEY_TO_JSON``). Counts: no_op 21, one_op 637, two_op 686,
    majority 343, choice 343. Each string compiles via :func:`rule_to_lut`.
    Strings are NOT deduped by LUT — a family with multiple string spellings of
    the same truth table keeps both, matching the uniform-over-strings measure
    the degeneracy / leakage analysis was built on."""
    S = _SHIFTS
    if key == "no_op":
        return [f"{f}{n}" for f in ("ROT", "SHL", "SHR") for n in S]
    if key == "one_op":
        return ([f"ROT{b} {op} SHL{a}" for b in S for a in S for op in _OPS6]
                + [f"ROT{b} {op} SHR{c}" for b in S for c in S for op in _OPS6]
                + [f"SHL{a} XOR SHR{c}" for a in S for c in S])
    if key == "two_op":
        return [s for a in S for b in S for c in S for s in
                (f"(SHL{a} XORNOT ROT{b}) OR SHR{c}",
                 f"(SHR{c} ORNOT ROT{b}) XOR SHL{a}")]
    if key == "majority":
        return [f"Maj(SHL{a}, ROT{b}, SHR{c})" for a in S for b in S for c in S]
    if key == "choice":
        return [f"Ch(SHL{a}, SHR{c}, ROT{b})" for a in S for b in S for c in S]
    raise ValueError(f"no grammar enumeration for family key {key!r}")


class BitManipulationGenerator(SyntheticGenerator):
    """Synthesize bit_manipulation problems by drawing from the rule catalog.

    Parameters
    ----------
    seed : int
        RNG seed.
    family_mix : Optional[Dict[str, float]]
        Family proportions for drawing from ``csvs/bit_manip_rules.json``.
        Keys must be a subset of
        ``{"no_op", "one_op", "two_op", "majority", "choice", "unknown"}``
        (the first five mapped to JSON families ``no-op / one-op / 2-op / maj /
        ch``). Within a family, rules are drawn uniformly. The special
        ``"unknown"`` key is NOT catalog-sourced: it generates clean Shape-C
        live-middle rows on the fly (see ``_UNKNOWN_KEY``). Weights are
        normalized. When ``None`` (and ``family_base_mix`` is also ``None``),
        the default ``_DEFAULT_FAMILY_MIX`` is used.
    family_base_mix : Optional[Dict[str, float]]
        Procedural mode (mutually exclusive with ``family_mix``). Same keys as
        ``family_mix``, but interpreted as BASE-family sampling weights: pick a
        base family, then draw a rule UNIFORMLY from that family's full grammar
        (shifts a,b,c ~ Uniform{1..7}) and emit it as-is — no catalog, no
        cascade resolution. Degenerate (a,b,c) collapse the function to a
        simpler family, so the RESOLVED mix the model sees drifts from these
        weights (e.g. ch leaks ~57% to one_op/no_op). The catalog is NOT read.
        ``"unknown"`` behaves as in ``family_mix`` (on-the-fly Shape-C). Pick
        the weights to land the drifted/resolved mix where you want it — see the
        documented degenerate output in final.yaml.
    rules_path : Optional[str]
        Override the default ``csvs/bit_manip_rules.json`` location. Consulted
        in catalog mode (the default, or when ``family_mix`` is provided).
    noise_p : float
        Deprecated and ignored — the ``"unknown"`` family is now the exact
        Shape-C rule, not noised 2-op. Still accepted so existing configs
        (e.g. final) that set it keep loading.
    """

    N_EXAMPLES_CHOICES: Tuple[int, ...] = (7, 8, 9, 10)

    def __init__(
        self,
        seed: int = 42,
        family_mix: Optional[Dict[str, float]] = None,
        family_base_mix: Optional[Dict[str, float]] = None,
        rules_path: Optional[str] = None,
        noise_p: float = _DEFAULT_NOISE_P,
        overrun_p: float = 0.0,
        **legacy: object,
    ) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)
        del noise_p                          # deprecated/ignored (see docstring)

        # Hard floor on the fraction of emitted rows whose solver path is the
        # block-1 "spurious over-run" fallback (`classify_fallback == 1`). 0.0 =
        # off: generate_one is byte-identical to the legacy body (no extra RNG
        # draws, no detector calls). Set before the mode-branch returns below so it
        # applies to catalog/procedural modes alike. See generate_one.
        self._overrun_p = float(overrun_p)
        self._emitted = 0                    # rows returned so far
        self._overrun_emitted = 0            # of those, confirmed block-1 rows

        ignored = [k for k in legacy if k in _LEGACY_KWARGS]
        if ignored:
            print(f"[bit_manipulation_syn] ignoring legacy kwargs "
                  f"{ignored} (rule-picker has no replay/oversample knobs)")
        unexpected = [k for k in legacy if k not in _LEGACY_KWARGS]
        if unexpected:
            raise TypeError(
                f"[bit_manipulation_syn] unexpected kwargs: {unexpected}")

        if family_mix is not None and family_base_mix is not None:
            raise ValueError(
                "set only one of family_mix (catalog) / family_base_mix "
                "(procedural)")

        if family_base_mix is not None:
            self._init_procedural(family_base_mix)
            return

        # Default and explicit family_mix both draw from the tracked catalog
        # csvs/bit_manip_rules.json; family_mix steers the per-family weights.
        self._init_from_rules_json(
            family_mix=dict(family_mix) if family_mix is not None
            else dict(_DEFAULT_FAMILY_MIX),
            rules_path=rules_path or _default_rules_path(),
        )

    def _init_from_rules_json(
        self,
        family_mix: Dict[str, float],
        rules_path: str,
    ) -> None:
        """Populate LUT buckets from ``csvs/bit_manip_rules.json``.

        Each entry contributes one LUT (compiled from its first ``rule``
        string, with the ``OP ~ATOM`` shorthand normalized) and is bucketed
        by its primary family (``entry["family"][0]``). Within a family
        weights are uniform (each rule is weight 1); the user-supplied
        ``family_mix`` only steers the family-level sampling.
        """
        allowed = set(_FAMILY_KEY_TO_JSON) | {_UNKNOWN_KEY}
        bad = set(family_mix) - allowed
        if bad:
            raise ValueError(
                f"unknown family_mix keys: {sorted(bad)} "
                f"(expected subset of {sorted(allowed)})"
            )

        with open(rules_path) as f:
            data = json.load(f)

        # Bucket by primary family (the first entry of the family list).
        json_to_key = {v: k for k, v in _FAMILY_KEY_TO_JSON.items()}
        cat_luts: Dict[str, List[np.ndarray]] = {k: [] for k in _FAMILY_KEY_TO_JSON}
        for rid, entry in data.items():
            fams = entry.get("family") or []
            rules = entry.get("rule") or []
            if not fams or not rules:
                continue
            key = json_to_key.get(fams[0])
            if key is None:
                continue
            try:
                lut = rule_to_lut(_normalize_rule_string(rules[0]))
            except ValueError:
                continue
            cat_luts[key].append(lut)

        empty = [k for k, v in cat_luts.items() if not v and family_mix.get(k, 0.0) > 0]
        if empty:
            raise RuntimeError(
                f"[bit_manipulation_syn] no rules parsed for families "
                f"{empty} in {rules_path}"
            )

        # Uniform cumulative weights within each family.
        self._cat_luts = cat_luts
        self._cat_cum = {
            k: [float(i + 1) for i in range(len(luts))]
            for k, luts in cat_luts.items()
        }
        self._cat_mass = {k: float(len(luts)) for k, luts in cat_luts.items()}
        self._pool = [lut for luts in cat_luts.values() for lut in luts]

        self._cat_names = [k for k in _FAMILY_KEY_TO_JSON if family_mix.get(k, 0.0) > 0.0]
        if family_mix.get(_UNKNOWN_KEY, 0.0) > 0.0:   # on-the-fly Shape-C rows
            self._cat_names.append(_UNKNOWN_KEY)
        if not self._cat_names:
            raise ValueError("family_mix has no positive-weight family")
        self._cat_probs = [float(family_mix[k]) for k in self._cat_names]

    def _init_procedural(self, family_base_mix: Dict[str, float]) -> None:
        """Populate LUT buckets by ENUMERATING each base family's grammar
        (uniform over rule strings), instead of reading the catalog.

        Keyed by base family; ``generate_one`` then samples a family by
        ``family_base_mix`` and a rule uniformly within it via the shared
        ``_sample_rule`` path. The emitted truth table is whatever that
        (a,b,c) produces — including degenerate collapses — so no resolution
        happens here (the drift is intrinsic; see the class docstring)."""
        allowed = set(_FAMILY_KEY_TO_JSON) | {_UNKNOWN_KEY}
        bad = set(family_base_mix) - allowed
        if bad:
            raise ValueError(
                f"unknown family_base_mix keys: {sorted(bad)} "
                f"(expected subset of {sorted(allowed)})")

        cat_luts: Dict[str, List[np.ndarray]] = {}
        for key in _FAMILY_KEY_TO_JSON:                 # excludes _UNKNOWN_KEY
            if family_base_mix.get(key, 0.0) <= 0.0:
                cat_luts[key] = []
                continue
            cat_luts[key] = [rule_to_lut(s) for s in _enumerate_family_rules(key)]

        self._cat_luts = cat_luts
        self._cat_cum = {
            k: [float(i + 1) for i in range(len(luts))]
            for k, luts in cat_luts.items()
        }
        self._cat_mass = {k: float(len(luts)) for k, luts in cat_luts.items()}
        self._pool = [lut for luts in cat_luts.values() for lut in luts]

        self._cat_names = [
            k for k in _FAMILY_KEY_TO_JSON if family_base_mix.get(k, 0.0) > 0.0]
        if family_base_mix.get(_UNKNOWN_KEY, 0.0) > 0.0:   # on-the-fly Shape-C
            self._cat_names.append(_UNKNOWN_KEY)
        if not self._cat_names:
            raise ValueError("family_base_mix has no positive-weight family")
        self._cat_probs = [float(family_base_mix[k]) for k in self._cat_names]

    def _sample_rule(self, cat: str) -> np.ndarray:
        cum = self._cat_cum[cat]
        target = self._rng.random() * cum[-1]
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return self._cat_luts[cat][lo]

    def _problem_from_lut(self, lut: np.ndarray,
                          n_ex_choices: Optional[Tuple[int, ...]] = None
                          ) -> Dict[str, str]:
        """Build one problem dict given an already-resolved 256-entry LUT.

        Shared by ``generate_one`` (LUT drawn from the catalog) and
        ``generate_from_rule`` (LUT compiled from a caller-supplied rule).
        Consumes the same RNG calls in the same order as the old inline
        body, so reproducibility is unchanged. ``n_ex_choices``
        overrides the example-count menu (the `overrun_p` forced path passes
        ``(7,)`` — fewer columns leave corner lengths under-determined, so the
        greedy longest run over-runs a boundary more often); ``None`` uses the
        default :data:`N_EXAMPLES_CHOICES`.
        """
        n_ex = self._rng.choice(n_ex_choices or self.N_EXAMPLES_CHOICES)
        inputs = self._rng.sample(range(256), n_ex + 1)
        outputs = [int(lut[i]) for i in inputs]

        pairs_str = "\n".join(
            f"{inputs[i]:08b} -> {outputs[i]:08b}" for i in range(n_ex)
        )
        prompt = _PROMPT_TEMPLATE.format(
            pairs=pairs_str,
            query=f"{inputs[n_ex]:08b}",
        )
        return {
            "id": make_id(),
            "prompt": prompt,
            "answer": f"{outputs[n_ex]:08b}",
        }

    def _generate_unknown(self) -> Dict[str, str]:
        """One "unknown" row: the clean Shape-C live-middle rule
        ``(SHLa XORNOT ROTb) OR SHRc`` with its all-ones cell cleared (see
        ``_gte_lut``). Re-drawn until the EXAMPLES distinguish Shape C from BOTH
        2-op grammar shapes (A and B) — some example bit must land on an all-ones
        middle cell, where C clears what A/B keep — so the examples genuinely
        require the new combiner and the row is grammar-unsolvable, not a plain
        2-op draw the solver would pick A or B for."""
        # Pick (a, b, c) whose Shape C is GLOBALLY distinct from both 2-op shapes.
        # Degenerate triples collapse C onto Shape B — b==a or b==8-c force T==L or
        # T==R, so the (1,0,1) cell that separates C from B is unreachable for ANY
        # input (~29% of a+c<=7 triples). Without this gate the example loop below,
        # which demands a C-vs-B divergence, would never terminate.
        while True:
            while True:                      # live middle requires a + c <= 7
                a, c = self._rng.randint(1, 7), self._rng.randint(1, 7)
                if a + c <= 7:
                    break
            b = self._rng.randint(1, 7)
            gte = _gte_lut(a, b, c)
            shape_a = rule_to_lut(f"(SHL{a} XORNOT ROT{b}) OR SHR{c}")
            shape_b = rule_to_lut(f"(SHR{c} ORNOT ROT{b}) XOR SHL{a}")
            if bool((gte != shape_a).any()) and bool((gte != shape_b).any()):
                break
        rule = (f"(SHL{a} XORNOT ROT{b}) OR SHR{c} "
                f"ANDNOT (SHL{a} AND ROT{b} AND SHR{c})")
        n_ex = self._rng.choice(self.N_EXAMPLES_CHOICES)
        while True:
            inputs = self._rng.sample(range(256), n_ex + 1)
            ex = inputs[:n_ex]
            if (any(gte[i] != shape_a[i] for i in ex)
                    and any(gte[i] != shape_b[i] for i in ex)):
                break
        outputs = [int(gte[i]) for i in inputs]
        pairs_str = "\n".join(
            f"{inputs[i]:08b} -> {outputs[i]:08b}" for i in range(n_ex)
        )
        prompt = _PROMPT_TEMPLATE.format(
            pairs=pairs_str, query=f"{inputs[n_ex]:08b}")
        return {
            "id": make_id(),
            "prompt": prompt,
            "answer": f"{outputs[n_ex]:08b}",
        }

    def _generate_normal(self) -> Dict[str, str]:
        """One row from the configured mix — the legacy `generate_one` body,
        RNG-draw-for-RNG-draw identical so `overrun_p == 0.0` reproduces old runs."""
        cat = self._rng.choices(self._cat_names, weights=self._cat_probs, k=1)[0]
        if cat == _UNKNOWN_KEY:
            return self._generate_unknown()
        lut = self._sample_rule(cat)
        return self._problem_from_lut(lut)

    def _is_overrun(self, row: Dict[str, str]) -> bool:
        """Whether `row`'s solver path is the block-1 over-run fallback. Parses the
        emitted prompt with the solver's own `_PAIR_RE` and runs the same
        `_render_and_select` the trainer runs (deterministic, no RNG) so the verdict
        can't drift from the rendered trace."""
        return classify_fallback(_PAIR_RE.findall(row["prompt"])) == 1

    def _generate_forced_overrun(self) -> Dict[str, str]:
        """Biased rejection sampling for a guaranteed block-1 row: draw only from
        live-middle families on the short (n_ex=7) menu, accept iff `_is_overrun`.
        The bias changes only the PROPOSAL distribution — acceptance stays exact —
        so the floor is honoured regardless. Fails loud if no hit in
        `_OVERRUN_MAX_TRIES` (a guarantee must not silently emit a non-over-run row)."""
        cats = [c for c in self._cat_names if c in _LIVE_MIDDLE_CATS]
        if not cats:
            raise RuntimeError(
                f"[bit_manipulation_syn] overrun_p={self._overrun_p} needs a "
                f"live-middle family (one of {sorted(_LIVE_MIDDLE_CATS)}) in the "
                f"mix, but cat_names={self._cat_names}")
        probs = [self._cat_probs[self._cat_names.index(c)] for c in cats]
        for _ in range(_OVERRUN_MAX_TRIES):
            cat = self._rng.choices(cats, weights=probs, k=1)[0]
            row = self._problem_from_lut(self._sample_rule(cat), n_ex_choices=(7,))
            if self._is_overrun(row):
                return row
        raise RuntimeError(
            f"[bit_manipulation_syn] overrun_p={self._overrun_p}: no block-1 row "
            f"in {_OVERRUN_MAX_TRIES} biased tries (achieved "
            f"{self._overrun_emitted}/{self._emitted}); check the family mix")

    def generate_one(self) -> Dict[str, str]:
        if self._overrun_p <= 0.0:           # knob off -> legacy path, untouched
            return self._generate_normal()
        # Deterministic largest-remainder floor: force a block-1 row exactly when the
        # achieved fraction would otherwise dip below `overrun_p` after this emission,
        # so overrun_emitted/emitted >= overrun_p holds for EVERY prefix. Naturally
        # occurring over-runs are counted too, so forcing only covers the shortfall.
        force = self._overrun_emitted < self._overrun_p * (self._emitted + 1)
        if force:
            row = self._generate_forced_overrun()
            self._overrun_emitted += 1
        else:
            row = self._generate_normal()
            if self._is_overrun(row):
                self._overrun_emitted += 1
        self._emitted += 1
        return row

    def generate_from_rule(self, rule: str) -> Dict[str, str]:
        """Synthesize one problem whose hidden transform is exactly ``rule``.

        Same surface format and example-count distribution as
        ``generate_one``, but the LUT is compiled from ``rule`` (via
        :func:`rule_to_lut`) instead of being drawn from the catalog. Raises
        ``ValueError`` if ``rule`` is not a parseable rule string.
        """
        lut = rule_to_lut(rule)
        row = self._problem_from_lut(lut)
        return row
