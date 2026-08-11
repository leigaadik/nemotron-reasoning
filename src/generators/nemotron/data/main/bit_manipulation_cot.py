"""See bit_manipulation_bit_formulas.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


NB = 8
_CONSTANTS: Tuple[str, ...] = ("LOW", "HIGH")
_SHIFTED_SECTIONS: Tuple[str, ...] = ("Identity", "NOT")
SYM_FAMILIES: Tuple[str, ...] = ("XOR", "OR", "AND", "XORNOT")
ASYM_FAMILIES: Tuple[str, ...] = ("ANDNOT", "ORNOT")
_BINARY_FAMILIES: Tuple[str, ...] = ("AND", "OR", "XOR", "ANDNOT", "ORNOT", "XORNOT")
UNARY_SECTIONS: Tuple[str, ...] = _CONSTANTS + _SHIFTED_SECTIONS
# Families that can fill the MIDDLE region: only LOW and NOT. No 3-region
# signature has HIGH or Identity as its middle (they appear only as corners), so
# HIGH/Identity are excluded from all middle-cycle work — survey, fillers, the
# per-family "Middle between" block, and the Middle-any table. They stay full
# participants in corner (Left/Right) matching.
MIDDLE_CYCLE_SECTIONS: Tuple[str, ...] = ("LOW", "NOT")
SECTION_ORDER: Tuple[str, ...] = UNARY_SECTIONS + _BINARY_FAMILIES
_DROP_ZERO_SHIFT: Tuple[str, ...] = _SHIFTED_SECTIONS + SYM_FAMILIES + ASYM_FAMILIES
MIDDLE_START = 1
MIDDLE_END = NB - 2  # 6: middle is bits 1..6, exclusive of edges 0, 7

_NO_SHARED_SHIFT_PAIRS = {("Identity", "Identity")}


# The Sig table is the single, self-describing source for region signatures:
# each entry declares its own family decomposition, so no token parsing is needed.
@dataclass(frozen=True)
class Sig:
    """One coalesced region signature, fully self-describing.

    `key` is the terse signature string (display id / lookup handle). `fams` is
    the per-region family decomposition (one family per region, so `len(fams)` is
    the region count); it replaces parsing the key. `recipe` names the
    `_recover_ac` extractor (see that function). `cut` is the 2-region cut section
    (`"8-a"`/`"c"`/`"empty"`; `""` for 1/3-region). `gate` is the regime rule,
    informational — only its PRESENCE (None vs not) gates band display, and only
    for 2-region; all 3-region bands are shown structurally."""
    key: str
    fams: Tuple[str, ...]
    recipe: str
    cut: str = ""
    gate: Optional[str] = None


# The signature table: each entry declares its family decomposition (`fams`,
# whose length is the region count), recovery recipe, cut section and regime
# gate. Laid out in WALK/DISPLAY order — 1-region, 2-region, then 3-region split
# by middle filler (LOW, then NOT, then live/unknown middles), matching the order
# `_walk` emits its sections (`_THREE` fillers run in `MIDDLE_CYCLE_SECTIONS`
# order: LOW, NOT). Region-count grouping is what `_ONE`/`_TWO`/`_THREE` read off;
# within a region count the relative order here is the display order. See docs
# ("Final summary E").
SIGNATURES: Tuple[Sig, ...] = (
    # 1-region. HIGH covers the all-ones output (the constant `ROTk ORNOT SHLk` /
    # `ROTk ORNOT SHR(8-k)` rules collapse to a single HIGH region spanning 8 bits).
    # AND covers a `Maj(...)` that collapses to a single `I_x AND I_y` spanning 8.
    Sig("HIGH", ("HIGH",), "oneregion"),
    Sig("Ib", ("Identity",), "oneregion"),
    Sig("ANDab", ("AND",), "oneregion"),

    # 2-region (cut set; gate only on the 4 noted collapse cases).
    Sig("LOW Ic", ("LOW", "Identity"), "const", "c"),
    Sig("LOW ANDbc", ("LOW", "AND"), "const", "c"),
    Sig("HIGH NOTb", ("HIGH", "NOT"), "const", "8-a"),
    Sig("HIGH ORNOTbc", ("HIGH", "ORNOT"), "const", "c"),
    Sig("HIGH ORNOTcb", ("HIGH", "ORNOT"), "const", "8-a"),
    Sig("Ia LOW", ("Identity", "LOW"), "const", "8-a"),
    Sig("Ia ANDbc", ("Identity", "AND"), "collapse_a_eq_b", "8-a", "negative"),
    Sig("Ib ORbc", ("Identity", "OR"), "shr_1op", "c"),
    Sig("Ib XORbc", ("Identity", "XOR"), "shr_1op", "c"),
    Sig("Ib ANDNOTbc", ("Identity", "ANDNOT"), "shr_1op", "c"),
    Sig("NOTb HIGH", ("NOT", "HIGH"), "const", "c"),
    Sig("NOTb XORNOTbc", ("NOT", "XORNOT"), "shr_1op", "c"),
    Sig("ANDab LOW", ("AND", "LOW"), "const", "8-a"),
    Sig("ANDab Ib", ("AND", "Identity"), "collapse_b_eq_c", "c", "negative"),
    Sig("ANDab ANDbc", ("AND", "AND"), "both_binary", "empty", "zero"),
    Sig("ORab Ib", ("OR", "Identity"), "shl_1op", "8-a"),
    Sig("XORab Ib", ("XOR", "Identity"), "shl_1op", "8-a"),
    Sig("ANDNOTba Ib", ("ANDNOT", "Identity"), "andnot_ib", "empty"),  # cut=empty places it in the walk's empty section; andnot_ib ignores sig.cut
    Sig("ORNOTba HIGH", ("ORNOT", "HIGH"), "const", "8-a"),
    Sig("XORNOTba NOTb", ("XORNOT", "NOT"), "shl_1op", "8-a"),
    Sig("XORNOTab ORNOTcb", ("XORNOT", "ORNOT"), "both_binary", "empty", "zero"),

    # 3-region (live middles dispatch to "multiop" via the MultiOp fast-path;
    # dead middles read off the corners). Gates are documentation only. Grouped by
    # middle filler in `MIDDLE_CYCLE_SECTIONS` order (LOW, NOT), then live/unknown.

    # LOW middle
    Sig("Ia LOW Ic", ("Identity", "LOW", "Identity"), "both_single", "", "positive"),
    Sig("Ia LOW ANDbc", ("Identity", "LOW", "AND"), "collapse_a_eq_b", "", "positive"),
    Sig("ANDab LOW Ib", ("AND", "LOW", "Identity"), "collapse_b_eq_c", "", "positive"),
    Sig("ANDab LOW ANDbc", ("AND", "LOW", "AND"), "both_binary", "", "positive"),

    # NOT middle
    Sig("HIGH NOTb ORNOTcb", ("HIGH", "NOT", "ORNOT"), "dead_high_ornot", "", "positive"),
    Sig("XORNOTab NOTa HIGH", ("XORNOT", "NOT", "HIGH"), "dead_xornot_high", "", "negative"),
    Sig("XORNOTab NOTb HIGH", ("XORNOT", "NOT", "HIGH"), "dead_xornot_high", "", "positive"),
    Sig("XORNOTab NOTb ORNOTcb", ("XORNOT", "NOT", "ORNOT"), "both_binary", "", "positive"),

    # Unknown (live) middle
    Sig("LOW ANDac Ib", ("LOW", "AND", "Identity"), "multiop", "", "negative"),
    Sig("HIGH NOT(ANDac) ORNOTcb", ("HIGH", "unknown", "ORNOT"), "multiop", "", "negative"),
    Sig("Ia XORac Ic", ("Identity", "XOR", "Identity"), "multiop", "", "negative"),
    Sig("ANDab MAJabc ANDbc", ("AND", "unknown", "AND"), "multiop", "", "negative"),
    Sig("ANDNOTba (ANDac)OR(ANDNOTba) Ib", ("ANDNOT", "unknown", "Identity"), "multiop", "", "negative"),
    Sig("XORNOTab ORNOTba HIGH", ("XORNOT", "ORNOT", "HIGH"), "multiop", "", "negative"),
    Sig("XORNOTab (ORNOTcb)XORIa ORNOTcb", ("XORNOT", "unknown", "ORNOT"), "multiop", "", "negative"),
    Sig("XORNOTab (XORNOTab)ORIc ORNOTcb", ("XORNOT", "unknown", "ORNOT"), "multiop", "", "negative"),
)
SIG_BY_KEY: Dict[str, Sig] = {s.key: s for s in SIGNATURES}
assert all(len(s.fams) == len(s.key.split()) for s in SIGNATURES), (
    "Sig.fams must carry one family per region (matching the key's token count)")


# Region-count groupings, read straight off the Sig table's `fams` (no parsing).
# Each element is the `(key, *fams)` tuple the walk consumes: `_ONE` -> (sig,
# fam); `_TWO` -> (sig, lfam, rfam); `_THREE` -> (sig, lfam, mfam, rfam).
_ONE: Tuple[Tuple[str, ...], ...] = tuple(
    (s.key, *s.fams) for s in SIGNATURES if len(s.fams) == 1)
_TWO: Tuple[Tuple[str, ...], ...] = tuple(
    (s.key, *s.fams) for s in SIGNATURES if len(s.fams) == 2)
_THREE: Tuple[Tuple[str, ...], ...] = tuple(
    (s.key, *s.fams) for s in SIGNATURES if len(s.fams) == 3)

# Family triple -> the signature key(s) sharing it. A 3-region layout can map to
# TWO signatures — the live (`a+c<=7`) and dead (`a+c>=9`) regime variants (e.g.
# `XORNOTab NOTa HIGH` vs `XORNOTab NOTb HIGH`). Keyed as a list so the regime is
# resolved by the actual `a+c` (see `_resolved_gate`) instead of by table order,
# which used to silently shadow one variant.
_SIG3_BY_TRIPLE: Dict[Tuple[str, ...], List[str]] = {}
for _s3 in SIGNATURES:
    if len(_s3.fams) == 3:
        _SIG3_BY_TRIPLE.setdefault(_s3.fams, []).append(_s3.key)


# 2-region cut sections: the single visible boundary is the SHL-death bit `8-a`,
# the SHR-onset bit `c`, or their coincidence `c=8-a` (empty middle, `a+c=8`).
# Each 2-region signature's `cut` field (the SIDE of its primary `(a, c)`
# candidate; see the SIGNATURES table) places it in one of these sections — used
# by `_walk` to group twins and by the `const` recipe to orient the cut.
_CUT_8A = "8-a"
_CUT_C = "c"
_CUT_EMPTY = "empty"          # cut at c = 8-a (a+c=8, no middle)
assert all(SIG_BY_KEY[sig].cut for sig, _l, _r in _TWO), (
    "every 2-region signature needs a cut side in the SIGNATURES table")


# 2-region walk order, grouped by the Final-summary-E length-candidate SHAPE
# (not by cut), so same-shape `pass true lengths` lines render adjacently:
#   1. `c, 8-c`   — single cut at c (SHR side)
#   2. `8-a, a`   — single cut at 8-a (SHL side)
#   3. `… / 8-b, b` — two candidates, b-triangulation (symmetric AND / Choice)
# The lone HIGH·ORNOT pair is split by its reading: `HIGH ORNOTbc` cuts at c
# (`(None,c)` → shape 1) and `HIGH ORNOTcb` at 8-a (`(a,None)` → shape 2), so each
# folds into the bucket whose length-shape it matches rather than a bucket of its own.
# Within each bucket signatures are in `_mirror_key` order (the canonical pair
# ordering used by `_three_pairs`): a corner pair and its left/right mirror share
# the same `(min,max)` family rank, so buckets 1 and 2 are MIRRORS of each other —
# every twin in `c,8-c` sits at the same relative slot as its reflection in
# `8-a,a` (LOW·Ic ↔ Ia·LOW, NOTb·HIGH ↔ HIGH·NOTb, HIGH·ORNOTbc ↔ ORNOTba·HIGH, the
# gated ANDab·Ib ↔ Ia·ANDbc, …). The reflection isn't a perfect bijection: a few
# twins have no in-bucket mirror — `Ib·ANDNOTbc`'s reflection is the two-cut
# `ANDNOTba·Ib` (bucket 3); `HIGH·ORNOTcb` (an ORNOT operand-order variant) and
# `XORNOTab·ORNOTcb` are `8-a,a`-only — so the columns drift by those extras.
# Selection is order-robust, so this is display-only.
_TWO_BUCKETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("c, 8-c", ("LOW Ic", "NOTb HIGH", "HIGH ORNOTbc", "ANDab Ib",
                "Ib ORbc", "Ib XORbc", "Ib ANDNOTbc", "NOTb XORNOTbc")),
    ("8-a, a", ("Ia LOW", "HIGH NOTb", "HIGH ORNOTcb", "ORNOTba HIGH", "Ia ANDbc",
                "ORab Ib", "XORab Ib", "XORNOTba NOTb", "XORNOTab ORNOTcb")),
    ("a/b two-cut", ("LOW ANDbc", "ANDab LOW", "ANDNOTba Ib", "ANDab ANDbc")),
)
assert {k for _lbl, _ks in _TWO_BUCKETS for k in _ks} == {t[0] for t in _TWO}, (
    "_TWO_BUCKETS must cover exactly the 2-region signatures (no missing/extra)")


# Section priority for mirror-ordering corner pairs (see `_mirror_key`).
_SECTION_RANK: Dict[str, int] = {f: i for i, f in enumerate(SECTION_ORDER)}


DISPLAY_MODE = "hybrid"

# EXPERIMENT: when True, the `Left right middle unknown` section is not tested
# at all if a prior section already supplied a passing option with `0 best`
# (the frozen seed is 0). Default off; behavior-neutral by construction (the
# strict `diff < seed` guard already tests nothing when seed==0).
_GATE_UNKNOWN_ON_ZERO_BEST = False

# When True, append a running per-length counter to each Left/Right/Middle chain
# line that shows " length N": the M-th chain of length N within that block
# renders " length N M". The counter is per block (Left, Right, Middle each
# start over) and counts only lines that show a length — dropped zero-shift
# rules (no " length" suffix) are skipped and don't advance the counter.
APPEND_LENGTH_COUNT = True


def _column_header(mode: str, n_examples: int) -> str:
    """Header line describing how an `n_examples`-long column signature renders.

    For hex/hybrid, spell out the 4-bit chunking: each full 4-bit group becomes
    one hex digit and the trailing short group stays binary under `hybrid`
    (e.g. 7 examples -> "7 examples 1 hex 3 binary") or is zero-padded to a hex
    digit under `hex`. `bits`/`bits_with_hash` keep their fixed labels."""
    if mode == "bits":
        return "Output bit columns."
    if mode == "bits_with_hash":
        return "Output bit signature with bitsum as hash"
    n_hex, rem = divmod(n_examples, 4)
    n_bin = rem if mode == "hybrid" else 0
    if mode == "hex" and rem:  # hex pads the short chunk up to a full hex digit
        n_hex += 1
    parts = [f"{n_examples} examples"]
    if n_hex:
        parts.append(f"{n_hex} hex")
    if n_bin:
        parts.append(f"{n_bin} binary")
    return "Output signature with " + " ".join(parts)

_PAIR_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QUERY_RE = re.compile(r"determine the output for:\s*([01]{8})", re.IGNORECASE)

# Region-placement step: (role, label, bits, note).
Step = Tuple[str, str, List[int], str]


@dataclass(frozen=True)
class ShiftRule:
    family: str
    shift_p: Optional[int]
    shift_s: Optional[int]

    @property
    def label(self) -> str:
        prefix = self.family
        if self.shift_p is None:
            return prefix
        if self.shift_s is None:
            return f"{prefix}{self.shift_p}"
        return f"{prefix}{self.shift_p}{self.shift_s}"

    def operands_at(self, bit: int) -> Tuple[Optional[int], Optional[int]]:
        p = (self.shift_p + bit) % NB if self.shift_p is not None else None
        s = (self.shift_s + bit) % NB if self.shift_s is not None else None
        return p, s


@dataclass(frozen=True)
class Record:
    family: str
    label: str
    primary: Optional[int]
    secondary: Optional[int]
    col: str
    matches: Tuple[int, ...]


@dataclass(frozen=True)
class OperandRecord:
    family: str
    primary: Optional[int]
    secondary: Optional[int]
    matches: Tuple[int, ...]


@dataclass(frozen=True)
class Columns:
    inputs: List[str]
    outputs: List[str]
    input_columns: List[str]
    output_columns: List[str]

    @property
    def n_examples(self) -> int:
        return len(self.inputs)


@dataclass
class MultiOp:
    family: str                  # "MAJ" | "Ch" | "2-op" | "XOR-shift"
    shl: Optional[int]
    rot: Optional[int]
    shr: Optional[int]
    shape: Optional[str] = None  # "A" | "B" | "C" (Shape A with its all-ones cell cleared)


BestChain = Tuple[ShiftRule, int]
Run = Tuple[ShiftRule, int, int, Optional[str]]  # (rule, start, length, failed)
CycleFill = Tuple[str, ShiftRule, int, int]      # (section, rule, start, length)


# Bit / column primitives.

def _normalize_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    return bits if len(bits) == NB else ""


def _bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def _invert(bits: str) -> str:
    return "".join(_bit_not(b) for b in bits)


def _column_bits(values: Sequence[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def _evaluate_binary(a: str, b: str, family: str) -> str:
    if family in ("AND", "ANDNOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "ORNOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XORNOT"):
        return "1" if a != b else "0"
    raise ValueError(f"Unsupported family {family}")


def _apply_family(a_bits: str, b_bits: str, family: str,
                  invert_second: bool = False) -> str:
    b_eff = _invert(b_bits) if invert_second else b_bits
    return "".join(_evaluate_binary(x, y, family) for x, y in zip(a_bits, b_eff))


def _build_columns(examples: Sequence[Tuple[str, str]]) -> Optional[Columns]:
    if not examples:
        return None
    inputs = [_normalize_bits(inp) for inp, _ in examples]
    outputs = [_normalize_bits(out) for _, out in examples]
    if any(not bits for bits in inputs + outputs):
        return None
    return Columns(
        inputs=inputs,
        outputs=outputs,
        input_columns=[_column_bits(inputs, i) for i in range(NB)],
        output_columns=[_column_bits(outputs, i) for i in range(NB)],
    )


def _build_records(cols: Columns) -> Dict[str, List[Record]]:
    output_columns = cols.output_columns
    input_columns = cols.input_columns
    input_inverted = [_invert(col) for col in input_columns]
    n = cols.n_examples

    def matches(col: str) -> Tuple[int, ...]:
        return tuple(i for i, oc in enumerate(output_columns) if col == oc)

    recs: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}

    for i, col in enumerate(input_columns):
        recs["Identity"].append(Record("I", str(i), i, None, col, matches(col)))
    for i, col in enumerate(input_inverted):
        recs["NOT"].append(Record("NOT", str(i), i, None, col, matches(col)))
    for val, section, family in (("0", "LOW", "LOW"), ("1", "HIGH", "HIGH")):
        col = val * n
        recs[section].append(Record(family, val, None, None, col, matches(col)))

    for fam in SYM_FAMILIES:
        invert = fam == "XORNOT"
        for circ_diff in range(1, NB // 2 + 1):
            n_pairs = NB // 2 if circ_diff == NB // 2 else NB
            for a in range(n_pairs):
                b = (a + circ_diff) % NB
                lo, hi = min(a, b), max(a, b)
                col = _apply_family(input_columns[lo], input_columns[hi], fam,
                                    invert_second=invert)
                recs[fam].append(Record(fam, f"{a}{b}", a, b, col, matches(col)))

    for fam in ASYM_FAMILIES:
        for diff in range(1, NB):
            for a in range(NB):
                b = (a + diff) % NB
                col = _apply_family(input_columns[a], input_columns[b], fam,
                                    invert_second=True)
                recs[fam].append(Record(fam, f"{a}{b}", a, b, col, matches(col)))

    for name in UNARY_SECTIONS:
        recs[name].sort(key=lambda r: r.label)
    return recs


def _expand_symmetric(records: Sequence[Record]) -> List[OperandRecord]:
    out: List[OperandRecord] = []
    for rec in records:
        out.append(OperandRecord(rec.family, rec.primary, rec.secondary, rec.matches))
        if rec.secondary is not None and rec.family in SYM_FAMILIES:
            out.append(OperandRecord(rec.family, rec.secondary, rec.primary, rec.matches))
    return out


def _shift_rule_at(orec: OperandRecord, bit: int) -> ShiftRule:
    sp = (orec.primary - bit) % NB if orec.primary is not None else None
    ss = (orec.secondary - bit) % NB if orec.secondary is not None else None
    return ShiftRule(orec.family, sp, ss)


def _candidates_at(orecords: Sequence[OperandRecord], bit: int) -> List[ShiftRule]:
    return [_shift_rule_at(o, bit) for o in orecords if bit in o.matches]


def _shift_matches(orecords: Sequence[OperandRecord]) -> Dict[ShiftRule, Tuple[int, ...]]:
    acc: Dict[ShiftRule, Set[int]] = {}
    for o in orecords:
        for bit in o.matches:
            acc.setdefault(_shift_rule_at(o, bit), set()).add(bit)
    return {rule: tuple(sorted(bits)) for rule, bits in acc.items()}


def _canon_key(family: str, p: Optional[int], s: Optional[int]) -> Tuple:
    if family in SYM_FAMILIES and p is not None and s is not None:
        return (family, min(p, s), max(p, s))
    return (family, p, s)


def _reproducing(orecords: Sequence[OperandRecord]
                 ) -> Set[Tuple[Optional[int], Optional[int]]]:
    """Operand pairs whose column reproduces at least one output bit."""
    return {(o.primary, o.secondary) for o in orecords if o.matches}


def _fail_expr(reproducing: Set[Tuple[Optional[int], Optional[int]]],
               ep: Optional[int], es: Optional[int]) -> Optional[str]:
    """The `<operands>x` token for a chain's failed next bit. 'x' = operands in
    the wrong position or not in the operator (the in-bounds run-end marker;
    out-of-bounds cycle ends use 'y', emitted in `_find_cycles`). Constants get
    none. ``reproducing`` is unused now that x/y are merged."""
    if ep is None:
        return None
    token = "x"  # "y" if (ep, es) in reproducing else "x"
    return f"{ep}{es}{token}" if es is not None else f"{ep}{token}"


def _ops_digits(rule: ShiftRule, bit: int) -> str:
    """Operand digits at `bit` in operand order; '' for constants."""
    p, s = rule.operands_at(bit)
    if p is None:
        return ""
    return str(p) if s is None else f"{p}{s}"


def _compact_ops(rule: ShiftRule, bit: int) -> str:
    """Operand digits at `bit` — canonical (sorted) for symmetric pairs."""
    p, s = rule.operands_at(bit)
    if p is None:                          # constant: show its output bit
        return "1" if rule.family == "HIGH" else "0"
    if s is None:
        return str(p)
    if rule.family in SYM_FAMILIES:
        return f"{min(p, s)}{max(p, s)}"
    return f"{p}{s}"


def _runs_from(orecords: Sequence[OperandRecord],
               matches: Dict[ShiftRule, Tuple[int, ...]],
               reproducing: Set[Tuple[Optional[int], Optional[int]]],
               anchor: int) -> List[Run]:
    """All maximal constant-shift runs anchored at output bit `anchor`
    (0 = Left, N-1 = Right), each with its failed-next-bit token."""
    runs: List[Run] = []
    seen: Set[Tuple] = set()
    for rule in _candidates_at(orecords, anchor):
        key = _canon_key(rule.family, *rule.operands_at(anchor))
        if key in seen:
            continue
        seen.add(key)
        bits = set(matches.get(rule, ()))
        length = 1
        if anchor == 0:
            while length < NB and length in bits:
                length += 1
            start = 0
            fail_bit = length if length < NB else None
        else:
            while length < NB and (NB - 1 - length) in bits:
                length += 1
            start = NB - length
            fail_bit = start - 1 if length < NB else None
        failed = (_fail_expr(reproducing, *rule.operands_at(fail_bit))
                  if fail_bit is not None else None)
        runs.append((rule, start, length, failed))
    return runs


def _find_cycles(orecords: Sequence[OperandRecord],
                 matches: Dict[ShiftRule, Tuple[int, ...]],
                 reproducing: Set[Tuple[Optional[int], Optional[int]]]
                 ) -> List[Run]:
    """Every maximal disjoint run of consecutive matching bits in the middle
    window for each rule, with the failed-next token ('x' within middle,
    'y' when the cycle would exit the middle)."""
    seen_rule: Set[ShiftRule] = set()
    ordered_rules: List[ShiftRule] = []
    for bit in range(MIDDLE_START, MIDDLE_END + 1):
        for rule in _candidates_at(orecords, bit):
            if rule not in seen_rule:
                seen_rule.add(rule)
                ordered_rules.append(rule)

    cycles: List[Run] = []
    for rule in ordered_rules:
        rule_bits = sorted(b for b in matches.get(rule, ())
                           if MIDDLE_START <= b <= MIDDLE_END)
        i = 0
        while i < len(rule_bits):
            start = rule_bits[i]
            length = 1
            while (i + length < len(rule_bits)
                   and rule_bits[i + length] == start + length):
                length += 1
            fb = start + length
            if fb <= MIDDLE_END:
                failed = _fail_expr(reproducing, *rule.operands_at(fb))
            else:
                p, s = rule.operands_at(fb)
                failed = (None if p is None
                          # else (f"{p}{s}z" if s is not None else f"{p}z"))
                          else (f"{p}{s}y" if s is not None else f"{p}y"))
            cycles.append((rule, start, length, failed))
            i += length

    rule_pos = {r: i for i, r in enumerate(ordered_rules)}
    cycles.sort(key=lambda c: (c[1], rule_pos.get(c[0], 1 << 30)))
    return cycles


def _rule_canon_key(rule: ShiftRule) -> Tuple:
    """Canonical dedup key for a rule; symmetric pairs collapse a/b <-> b/a."""
    return _canon_key(rule.family, rule.shift_p, rule.shift_s)


def _is_zero_shift(rule: ShiftRule) -> bool:
    """True when a displayed shift digit is 0 — an operand reads its
    output-aligned column. Constants (shift_p is None) are never zero-shift;
    for unary rules shift_s is None and is ignored."""
    return rule.shift_p == 0 or rule.shift_s == 0


def _tied_longest(runs: Sequence[Run]) -> List[BestChain]:
    if not runs:
        return []
    longest = max(length for _, _, length, _ in runs)
    return [(rule, length) for rule, _s, length, _f in runs if length == longest]


def _all_corner_runs(an: "Analysis", family: str, anchor: int) -> List[Run]:
    """Every maximal run of `family` anchored at `anchor` (0=Left, NB-1=Right),
    INCLUDING the sub-maximal runs `_tied_longest` discards. Recomputes the same
    `(orecords, matches, reproducing)` inputs `_emit_section` builds, so the
    no-winner fallback can retry the chain with a shorter, gate-consistent corner
    (greedy detection keeps only the longest, which can over-run a boundary)."""
    records = _build_records(an.cols).get(family, [])
    if not records:
        return []
    orec = _expand_symmetric(records)
    return _runs_from(orec, _shift_matches(orec), _reproducing(orec), anchor)


def _best_length(bests: Sequence[BestChain]) -> int:
    return bests[0][1] if bests else 0


# Middle layout, cycle survey, middle-setter shift detection.

def _middle_set(left_longest: int, right_longest: int) -> Set[int]:
    covered = list(range(left_longest))
    for bit in (NB - 1 - i for i in range(right_longest)):
        if bit not in covered:
            covered.append(bit)
    return set(range(NB)) - set(covered)


def _cycle_survey(section_cycles: Dict[str, List[Run]],
                  middle_set: Set[int]) -> List[CycleFill]:
    fillers: List[CycleFill] = []
    for name in MIDDLE_CYCLE_SECTIONS:
        for rule, start, length, _ in section_cycles.get(name, []):
            span = {(start + i) % NB for i in range(length)}
            if middle_set <= span:
                fillers.append((name, rule, start, length))
    return fillers


# Shift-match gate — the shared-shift test the signature walk applies inline
# (which Left/Right best chains keep, given a section/filler shift).

def _ml_match_keys(rule: ShiftRule, side: str) -> Set[object]:
    if rule.shift_p is None:
        return {None}
    if rule.shift_s is None:
        return {rule.shift_p}
    if rule.family in ("ORNOT", "ANDNOT"):
        return {rule.shift_p} if side == "Left" else {rule.shift_s}
    return {rule.shift_p, rule.shift_s}


def _filter_match(rule_a: ShiftRule, side_a: str,
                  rule_b: ShiftRule, side_b: str) -> bool:
    keys_a = _ml_match_keys(rule_a, side_a)
    keys_b = _ml_match_keys(rule_b, side_b)
    if None in keys_a or None in keys_b:
        return True
    return bool(keys_a & keys_b)


# DISPLAY / NARRATION up to middle filler section


def _column_hash(bits: str, total_examples: int) -> str:
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def _format_signature(bits: str, mode: str = DISPLAY_MODE) -> str:
    """Render a column signature for display under `mode`."""
    chunks = [bits[i:i + 4] for i in range(0, len(bits), 4)]
    parts: List[str] = []
    for idx, chunk in enumerate(chunks):
        is_last_short = idx == len(chunks) - 1 and len(chunk) < 4
        if mode in ("bits", "bits_with_hash"):
            parts.append(chunk)
        elif mode == "hex":
            parts.append(format(int(chunk.ljust(4, "0"), 2), "X"))
        elif mode == "hybrid":
            if is_last_short:
                parts.append((" " if parts else "") + chunk)  # space off hex run
            else:
                h = format(int(chunk, 2), "X")
                parts.append(f" {h}" if h.isalpha() else h)   # A-F: preceding space
    if mode == "hybrid":
        out = "".join(parts)                 # no-space join; digit runs pack up
    else:
        out = " ".join(parts)
    if mode == "bits_with_hash":
        out += " " + _column_hash(bits, len(bits))
    return out


def _group_bits(s: str, group: int = 4) -> str:
    """Insert a space every `group` characters (the raw-bit grouping the Testing
    gap-evaluation lines use, matching the old CoT Apply rendering)."""
    return " ".join(s[i:i + group] for i in range(0, len(s), group))


def _emit_column_section(direction: str, columns: Sequence[str],
                         mode: str = DISPLAY_MODE) -> List[str]:
    """The per-direction column block (raw bits block first for hybrid/hex)."""
    section_modes = ("bits", mode) if mode in ("hex", "hybrid") else (mode,)
    n_examples = len(columns[0]) if columns else 0
    lines: List[str] = []
    for sec_mode in section_modes:
        base = _column_header(sec_mode, n_examples)
        lines.append(base if direction == "Output"
                     else base.replace("Output", direction))
        lines.append("")
        for bit in range(NB):
            lines.append(f"{bit}│{_format_signature(columns[bit], sec_mode)}")
        lines.append("")
    return lines


def _emit_example_block(lines: List[str], direction: str,
                        examples: Sequence[str]) -> None:
    """The per-example `<direction> {i}` header + bit│val rows."""
    for i, bits in enumerate(examples):
        lines.append(f"{direction} {i}")
        for bit in range(NB):
            lines.append(f"{bit}│{bits[bit]}")
        lines.append("")


def _chain_items(rule: ShiftRule, bits_seq: Sequence[int],
                 fail_bit: Optional[int], fail_token: Optional[str]) -> str:
    parts: List[str] = []
    for b in bits_seq:
        digits = _ops_digits(rule, b)
        parts.append(f"{b} {digits}" if digits else str(b))
    if fail_bit is not None and fail_token:
        parts.append(f"{fail_bit} {fail_token}")
    return "│".join(parts)


def _length_suffix(length: int, with_length: bool, count: Optional[int]) -> str:
    """`" length N"`, plus `" M"` when a running counter is supplied (only when
    the length is shown)."""
    if not with_length:
        return ""
    suffix = f" length {length}"
    if count is not None:
        suffix += f" {count}"
    return suffix


def _render_chain_line(rule: ShiftRule, bits_seq: Sequence[int],
                       fail_bit: Optional[int], fail_token: Optional[str],
                       with_length: bool = True,
                       count: Optional[int] = None) -> str:
    items = _chain_items(rule, bits_seq, fail_bit, fail_token)
    suffix = _length_suffix(len(bits_seq), with_length, count)
    return f"{items} {rule.label}{suffix}"


def _render_cycle_line(rule: ShiftRule, start: int, length: int,
                       fail_token: Optional[str], with_length: bool = True,
                       count: Optional[int] = None) -> str:
    seq = [(start + i) % NB for i in range(length)]
    fail_bit = (start + length) % NB if fail_token else None
    items = _chain_items(rule, seq, fail_bit, fail_token)
    ident = _cycle_ident(rule, start, length)
    suffix = _length_suffix(length, with_length, count)
    return f"{items} {ident}{suffix}"


def _dedup_by_canon(rules: Iterable[ShiftRule]) -> List[ShiftRule]:
    """Rules in order, keeping the first occurrence of each canonical key."""
    seen: Set[Tuple] = set()
    out: List[ShiftRule] = []
    for rule in rules:
        key = _rule_canon_key(rule)
        if key not in seen:
            seen.add(key)
            out.append(rule)
    return out


def _render_best_section(bests: Sequence[BestChain], label: str = "Best") -> str:
    if not bests:
        return f"{label} none"
    starters = [r.label for r in _dedup_by_canon(r for r, _ in bests)]
    length = _best_length(bests)
    if APPEND_LENGTH_COUNT:
        # Best length, count of best chains at that length, then each starter
        # prefixed with its 1-based index (matching the chain lines' "length N M").
        enumerated = " ".join(f"{i} {s}" for i, s in enumerate(starters, 1))
        return f"{label} {length} {len(starters)} {enumerated}"
    return f"{label} {length} " + " ".join(starters)


def _render_any_cycle(cycles: Sequence[Run], label: str = "Any") -> str:
    """List every matching middle cycle (spanning or not), not just the
    longest — the middle keeps all candidates rather than picking a best."""
    if not cycles:
        return f"{label} none"
    return f"{label} " + " ".join(_cycle_ident(r, s, l) for r, s, l, _ in cycles)


def _emit_block(lines: List[str], header: str, items: Sequence[str],
                best_line: str) -> None:
    lines.append(header)
    lines.extend(items if items else ["none"])
    lines.append(best_line)
    lines.append("")


@dataclass(frozen=True)
class _SectionResult:
    left_bests: List[BestChain]
    right_bests: List[BestChain]
    cycles: Optional[List[Run]]


def _emit_section(name: str, records: Sequence[Record],
                  lines: List[str]) -> _SectionResult:
    """Emit one operation section: raw signatures, per-bit matches, the
    Left/Right chain summaries, and (for unary sections) the Cycle block.
    Appends to `lines` and returns the section's structured deliverable."""
    orecords = _expand_symmetric(records)
    matches = _shift_matches(orecords)
    reproducing = _reproducing(orecords)

    # Raw data — "{label}│{signature}" with optional " match …"; pair ops get a
    # blank line between circular-difference groups.
    lines.append(name)
    prev_diff: Optional[int] = None
    for rec in records:
        if (len(rec.label) >= 2 and rec.label[0].isdigit()
                and rec.label[1].isdigit()):
            diff = (int(rec.label[1]) - int(rec.label[0])) % NB
            if prev_diff is not None and diff != prev_diff:
                lines.append("")
            prev_diff = diff
        line = f"{rec.label}│{_format_signature(rec.col)}"
        if rec.matches:
            line += " match " + " ".join(str(i) for i in rec.matches)
        lines.append(line)
    lines.append("")

    # Per-bit matches — symmetric pairs collapse (a,b)↔(b,a) to one entry.
    lines.append(f"{name} output matches")
    for i in range(NB):
        display = _dedup_by_canon(_candidates_at(orecords, i))
        if display:
            lines.append(f"{i}│" + " ".join(_compact_ops(r, i) for r in display))
        else:
            lines.append(f"{i}│ absent")
    lines.append("")

    # Identity/NOT and every binary family drop zero-shift rules: the line is
    # still shown (sans " length N") but excluded from the best line (relabeled
    # "Best nonzero") and from predict()'s deliverable. LOW/HIGH (bare constants)
    # are unaffected.
    drop_zero = name in _DROP_ZERO_SHIFT
    best_label = "Best nonzero" if drop_zero else "Best"

    def _counts(rule: ShiftRule) -> bool:
        return not (drop_zero and _is_zero_shift(rule))

    # Per-block running counter of "how many chains of this length so far"
    # (None unless APPEND_LENGTH_COUNT is on, or for lines without a length).
    def _next_count(counter: Dict[int, int], length: int, counts: bool
                    ) -> Optional[int]:
        if not (APPEND_LENGTH_COUNT and counts):
            return None
        counter[length] = counter.get(length, 0) + 1
        return counter[length]

    # Left / Right chain summaries.
    left_runs = _runs_from(orecords, matches, reproducing, 0)
    right_runs = _runs_from(orecords, matches, reproducing, NB - 1)
    left_lines: List[str] = []
    left_counter: Dict[int, int] = {}
    for rule, _start, length, failed in left_runs:
        seq = list(range(length))
        fail_bit = length if failed else None
        counts = _counts(rule)
        count = _next_count(left_counter, len(seq), counts)
        left_lines.append(_render_chain_line(rule, seq, fail_bit, failed, counts, count))
    right_lines: List[str] = []
    right_counter: Dict[int, int] = {}
    for rule, start, length, failed in right_runs:
        seq = list(range(NB - 1, start - 1, -1))
        fail_bit = start - 1 if failed else None
        counts = _counts(rule)
        count = _next_count(right_counter, len(seq), counts)
        right_lines.append(_render_chain_line(rule, seq, fail_bit, failed, counts, count))

    left_bests = _tied_longest([r for r in left_runs if _counts(r[0])])
    right_bests = _tied_longest([r for r in right_runs if _counts(r[0])])
    _emit_block(lines, "Left", left_lines, _render_best_section(left_bests, best_label))
    _emit_block(lines, "Right", right_lines, _render_best_section(right_bests, best_label))

    cycles: Optional[List[Run]] = None
    if name in MIDDLE_CYCLE_SECTIONS:
        all_cycles = _find_cycles(orecords, matches, reproducing)
        cycle_counter: Dict[int, int] = {}
        cycle_lines = []
        for r, sb, length, failed in all_cycles:
            counts = _counts(r)
            count = _next_count(cycle_counter, length, counts)
            cycle_lines.append(
                _render_cycle_line(r, sb, length, failed, counts, count))
        # Deliverable + best line use only the nonzero cycles; the display
        # above still lists the dropped zero-shift cycles (sans length).
        cycles = [c for c in all_cycles if _counts(c[0])]
        header = f"Middle between {MIDDLE_START} to {MIDDLE_END}"
        any_label = "Any nonzero" if drop_zero else "Any"
        _emit_block(lines, header, cycle_lines, _render_any_cycle(cycles, any_label))

    return _SectionResult(left_bests, right_bests, cycles)


def _bit_list(seq: Sequence[int]) -> str:
    """Bit-index sequence, space-separated, order verbatim (a descending
    right-side sequence renders "7 6 5 4 3 2"). Empty -> "none"."""
    return " ".join(str(b) for b in seq) if seq else "none"


def _emit_cycle_survey(lines: List[str], section_cycles: Dict[str, List[Run]],
                       middle_set: Set[int]) -> List[CycleFill]:
    """Every unary cycle classified full|partial / secure|insecure, then the
    'Middle fillers are …' verdict listing every full cycle."""
    fillers: List[CycleFill] = []
    cycle_lines: List[str] = []
    for name in MIDDLE_CYCLE_SECTIONS:
        cycles = section_cycles.get(name, [])
        if not cycles:
            continue
        by_rule: List[Tuple[ShiftRule, List[Run]]] = []
        rule_idx: Dict[ShiftRule, int] = {}
        for run in cycles:
            r = run[0]
            if r not in rule_idx:
                rule_idx[r] = len(by_rule)
                by_rule.append((r, []))
            by_rule[rule_idx[r]][1].append(run)
        for rule, rule_runs in by_rule:
            shift = "any" if rule.shift_p is None else str(rule.shift_p)
            for _idx, (r, start, length, _failed) in enumerate(rule_runs):
                ident = _cycle_ident(r, start, length)
                span = [(start + i) % NB for i in range(length)]
                covers_middle = middle_set <= set(span)
                fill_kind = "spans" if covers_middle else "not"
                covers = f"{span[0]} to {span[-1]}"
                cycle_lines.append(
                    f"{ident} shift {shift} covers {covers} {fill_kind}")
                if covers_middle:
                    fillers.append((name, r, start, length))

    lines.extend(cycle_lines if cycle_lines else ["none"])
    def _filler_token(s: CycleFill) -> str:
        _, r, st, ln = s
        return _cycle_ident(r, st, ln)
    if fillers:
        middle_fillers = " ".join(_filler_token(s) for s in fillers)
        lines.append(f"Middle fillers span and are {middle_fillers} so test each")
    else:
        lines.append("Middle fillers span and are none so test none")
    return fillers


def _middle_best_block(section_cycles: Dict[str, List[Run]], left_longest: int,
                       right_longest: int) -> List[str]:
    """The 'Middle best from cycles' block: bracket the bits, survey the gap
    cycles. Shift derivation moved into the signature walk (each filler section
    states its own shift), so this block is narration only."""
    lines: List[str] = [""]
    middle_set = _middle_set(left_longest, right_longest)
    # No-middle gate: a two-region (no-middle) cut can tile all NB bits only if
    # the longest corner runs jointly reach NB, so it is tested iff
    # `left+right >= NB` (mirrors the `two` section gate in `_walk`).
    sum_longest = left_longest + right_longest
    verdict = "test" if sum_longest >= NB else "skip"
    lines.append(f"Test no middle if sum longest is at least {NB}")
    lines.append(f"Left longest {left_longest} right longest {right_longest} "
                 f"sum longest {sum_longest} so {verdict}")
    left_bits = list(range(left_longest))
    right_bits = [NB - 1 - i for i in range(right_longest)]
    lines.append(f"Left longest covers {_bit_list(left_bits)}")
    lines.append(f"Right longest covers {_bit_list(right_bits)}")
    lines.append(f"Middle remaining is {_bit_list(sorted(middle_set))}")
    lines.append("Middle either spans remaining or not")
    _emit_cycle_survey(lines, section_cycles, middle_set)
    return lines


# Opening analysis — `analyze` runs the whole up-to-middle-filler pipeline,
# the bridge between narration (`trace_lines`) and the structured `Analysis`.

@dataclass(frozen=True)
class Analysis:
    cols: Columns
    section_lefts: List[Tuple[str, List[BestChain]]]
    section_rights: List[Tuple[str, List[BestChain]]]
    section_cycles: Dict[str, List[Run]]
    left_longest: int
    right_longest: int
    trace_lines: List[str]   # the gold narration up through the middle-filler block


def analyze(examples: Sequence[Tuple[str, str]]) -> Optional[Analysis]:
    """Run the opening analysis: emit the preamble + per-section matching, the
    Left/Right/Middle best tables, and the middle-filler/shift section,
    capturing both the narration (`trace_lines`) and the structured deliverables
    the logic helpers and signature walk consume."""
    cols = _build_columns(examples)
    assert cols, "Columns could not be built."
    recs = _build_records(cols)

    lines: List[str] = [
        "We need to deduce the transformation by matching the example "
        "outputs. Let's see outputs.", "",
    ]
    _emit_example_block(lines, "Output", cols.outputs)
    lines.extend(_emit_column_section("Output", cols.output_columns))
    _emit_example_block(lines, "Input", cols.inputs)
    lines.extend(_emit_column_section("Input", cols.input_columns))

    lines.append("Now we need to compute output signatures from input "
                 "signatures and match them")
    and_count = " and count" if APPEND_LENGTH_COUNT else ""
    lines.append(f"format = input hex signatures│output signature from "
                 f"operation (match{and_count} if equal to output)")
    lines.append("x = wrong position or not in operator")
    lines.append("y = out of bounds")
    lines.append("")

    section_lefts: List[Tuple[str, List[BestChain]]] = []
    section_rights: List[Tuple[str, List[BestChain]]] = []
    section_cycles: Dict[str, List[Run]] = {}
    for name in SECTION_ORDER:
        result = _emit_section(name, recs[name], lines)
        section_lefts.append((name, result.left_bests))
        section_rights.append((name, result.right_bests))
        if result.cycles is not None:
            section_cycles[name] = result.cycles

    left_longest = max((_best_length(b) for _, b in section_lefts), default=0)
    right_longest = max((_best_length(b) for _, b in section_rights), default=0)
    lines += _best_table("Left", section_lefts, left_longest)
    lines.append("")
    lines += _best_table("Right", section_rights, right_longest)
    lines.append("")
    lines += _middle_any_table(section_cycles)

    lines += _middle_best_block(section_cycles, left_longest, right_longest)
    # Middle-left/right filtering moved into the signature walk: each section
    # states its own thresholds/shift and filters Left/Right best inline (length
    # logic at the family header, shift logic on the operator lines).

    return Analysis(
        cols, section_lefts, section_rights, section_cycles,
        left_longest, right_longest, lines,
    )


# PURE LOGIC (cont.) — the decision machinery the signature walk and apply step
# lean on: multi-op evaluation, the signature table, the observed-signature
# matcher, (a,c) recovery, and region placement. Rendered by the DISPLAY layer
# that follows.

# Multi-op evaluation (cf. `_multiop_eval_*`, `_apply_rule_to_query`).

def _atom_column_wrap(shift: Optional[int], bit: int,
                      input_columns: Sequence[str], n: int) -> str:
    if shift is None:
        return "0" * n
    return input_columns[(shift + bit) % NB]


def _multiop_eval_bit(family: str, shape: Optional[str],
                      left: int, rot: int, right: int) -> int:
    if family == "MAJ":
        return 1 if left + rot + right >= 2 else 0
    if family == "Ch":
        return (left & right) | (rot & (1 - left))
    if family == "XOR-shift":
        return left ^ right
    if family == "AND-shift":              # LOW/Identity middle ANDac
        return left & right
    if family == "NOTAND-shift":           # HIGH/ORNOT middle NOT(ANDac)
        return 1 - (left & right)
    if family == "ORNOT-shift":            # XORNOT/HIGH middle ORNOTxy = x|~y
        return (left | (1 - right)) & 1
    if shape == "B":
        return ((right | (1 - rot)) ^ left) & 1
    if shape == "C":                       # SHR-select: R ? NAND(L,T) : XNOR(L,T)
        return ((1 - (left & rot)) if right else (1 - (left ^ rot))) & 1
    return ((1 if left == rot else 0) | right) & 1


def _multiop_eval_column(mop: MultiOp, bit: int,
                         input_columns: Sequence[str], n: int) -> str:
    lc = _atom_column_wrap(mop.shl, bit, input_columns, n)
    tc = _atom_column_wrap(mop.rot, bit, input_columns, n)
    rc = _atom_column_wrap(mop.shr, bit, input_columns, n)
    return "".join(
        str(_multiop_eval_bit(mop.family, mop.shape, int(a), int(b), int(c)))
        for a, b, c in zip(lc, tc, rc)
    )


def _multiop_eval_query(mop: MultiOp, bit: int, query: str) -> str:
    """Single-bit apply: the column evaluator over a one-row column set."""
    return _multiop_eval_column(mop, bit, list(query), 1)


def _multiop_eval_raw(family: str, shape: Optional[str],
                      lc: str, tc: str, rc: str) -> str:
    """Evaluate a MultiOp over already-resolved equal-length operand bit strings
    (one char per example). The canonical result the Apply/Testing lines box."""
    return "".join(
        str(_multiop_eval_bit(family, shape, int(a), int(b), int(c)))
        for a, b, c in zip(lc, tc, rc)
    )


def _bits_zip(a: str, b: str, f) -> str:
    return "".join("1" if f(x == "1", y == "1") else "0" for x, y in zip(a, b))


def _multiop_rhs(family: str, shape: Optional[str],
                 l_in: str, t_in: str, r_in: str,
                 lc: str, tc: str, rc: str) -> str:
    """The shared RHS of a MultiOp derivation — the ``<fn> = <fn vals> = … =
    <final>`` chain AFTER the ``<label> = `` prefix — used by BOTH the single-bit
    Apply renderer and the column-vector Testing renderer so an operator reads
    identically in both blocks. ``l_in/t_in/r_in`` are the operand labels
    (``IN<col>`` or ``0`` for an absent atom); ``lc/tc/rc`` are the matching raw
    operand bit strings (length 1 for Apply, n_examples for Testing). The 2-op
    shapes expand to their primitive reduction (``XORNOT(L,T) OR R`` /
    ``L XOR OR(NOT T,R)``); ``NOTAND-shift`` shows its ``NOT AND`` step — the
    forms the model otherwise had to recover from the opaque ``XNO(a,b,c)``."""
    g = _group_bits
    lv, tv, rv = g(lc), g(tc), g(rc)
    final = g(_multiop_eval_raw(family, shape, lc, tc, rc))
    if family == "XOR-shift":                      # SHLa XOR SHRc (no rot)
        return f"{l_in} XOR {r_in} = {lv} XOR {rv} = {final}"
    if family in ("MAJ", "Ch"):
        gate = "Majority" if family == "MAJ" else "Choice"
        return (f"{gate}({l_in}, {t_in}, {r_in}) = "
                f"{gate}({lv}, {tv}, {rv}) = {final}")
    if family == "NOTAND-shift":                    # NOT(ANDac); expand the NOT step
        andc = _bits_zip(lc, rc, lambda x, y: x and y)
        return (f"¬ ({l_in} AND {r_in}) = ¬ ({lv} AND {rv}) = "
                f"¬ {g(andc)} = {final}")
    if family == "AND-shift":                       # LOW/Identity middle ANDac
        return f"{l_in} AND {r_in} = {lv} AND {rv} = {final}"
    if family == "ORNOT-shift":                      # XORNOT/HIGH middle ORNOTxy = x|¬y
        return f"{l_in} OR ¬ {r_in} = {lv} OR ¬ {rv} = {final}"
    # 2-op: shape A = (L XOR ¬T) OR R = XNO; shape B = L XOR (¬T OR R) = ONX.
    if shape == "C":
        # Shape C = GTE: Shape A with its all-ones middle cell cleared, i.e.
        # ((L XOR ¬T) OR R) AND ¬(L AND T AND R) — equivalently R selects
        # NAND(L, T) over XNOR(L, T). The ANDNOT term zeroes exactly the (1,1,1)
        # bits where it parts from Shape A, the structure the examples expose.
        xnorc = _bits_zip(lc, tc, lambda x, y: x == y)        # (L XOR ¬T)
        shapeac = _bits_zip(xnorc, rc, lambda x, y: x or y)   # (L XOR ¬T) OR R
        and3c = _bits_zip(_bits_zip(lc, tc, lambda x, y: x and y), rc,
                          lambda x, y: x and y)               # L AND T AND R
        return (f"(({l_in} XOR ¬ {t_in}) OR {r_in}) AND ¬ "
                f"({l_in} AND {t_in} AND {r_in}) = "
                f"(({lv} XOR ¬ {tv}) OR {r_in}) AND ¬ "
                f"({l_in} AND {t_in} AND {r_in}) = "
                f"({g(xnorc)} OR {rv}) AND ¬ "           # XOR group reduced; R→value
                f"({lv} AND {tv} AND {rv}) = "
                f"{g(shapeac)} AND ¬ "                   # ((L XOR ¬T) OR R) reduced
                f"{g(and3c)} = "
                f"{final}"
        )
    if shape == "B":
        # L XOR (¬T OR R). Keep the T-before-R order so the first line matches
        # the l,t,r ordering used by shapes A and C.
        innerc = _bits_zip(tc, rc, lambda t, r: (not t) or r)  # ¬T OR R
        return (f"{l_in} XOR (¬ {t_in} OR {r_in}) = "
                f"{l_in} XOR (¬ {tv} OR {rv}) = "            # L stays IN until the OR group reduces
                f"{lv} XOR {g(innerc)} = {final}")
    innerc = _bits_zip(lc, tc, lambda x, y: x == y)          # XORNOT(L, T) = L XOR ¬T
    return (f"({l_in} XOR ¬ {t_in}) OR {r_in} = "
            f"({lv} XOR ¬ {tv}) OR {r_in} = "                # R stays IN until the XOR group reduces
            f"{g(innerc)} OR {rv} = {final}")


def _apply_rule_to_query(rule: ShiftRule, bit: int, query: str) -> str:
    p, s = rule.operands_at(bit)
    fam = rule.family
    if fam == "LOW":
        return "0"
    if fam == "HIGH":
        return "1"
    if fam == "I":
        return query[p]
    if fam == "NOT":
        return _bit_not(query[p])
    a, b = query[p], query[s]
    if fam in ("ANDNOT", "ORNOT", "XORNOT"):
        b = _bit_not(b)
    return _evaluate_binary(a, b, fam)


def _apply_to_query(entry: object, bit: int, query: str) -> str:
    """Apply one per-bit plan entry to the query. None -> '0'."""
    if isinstance(entry, MultiOp):
        return _multiop_eval_query(entry, bit, query)
    if isinstance(entry, ShiftRule):
        return _apply_rule_to_query(entry, bit, query)
    assert False, "Every rule should map"
    # return "0"


# DISPLAY / NARRATION (cont.) — the signature walk. No early exit: run the full
# motor (best chains, middle filtering) and list every matching signature.
# Modelled on the shipped CoT's "Left best" / "Right best" / "Middle best" +
# filter blocks, then the candidate listing in place of the test driver. Like
# `analyze`, `_walk`/`_emit_testing` emit narration *and* return the
# structured `Possibility` picks the apply step consumes; `explain` stitches the
# whole trace together.

def _cycle_ident(rule: ShiftRule, start: int, length: int) -> str:
    shift = "a" if rule.shift_p is None else ""
    return f"{rule.label}{shift}{start}{start + length - 1}"


def _or_absent(name: str, items: Sequence[str]) -> str:
    """``<name> <items…>`` joined, or ``<name> absent`` when items is empty."""
    return " ".join([name, *items]) if items else f"{name} absent"


def _best_table(side: str, entries, longest: int) -> List[str]:
    out = [f"{side} best"]
    for name, bests in entries:
        n = _best_length(bests)
        items = [str(n), *(r.label for r, _ in bests)] if n else []
        out.append(_or_absent(name, items))
    out.append(f"{side} longest is {longest}")
    return out


def _middle_any_table(section_cycles) -> List[str]:
    """Per family, list every matching middle cycle (no length, no best pick);
    families with no cycle render 'absent'."""
    out = ["Middle any"]
    for name in MIDDLE_CYCLE_SECTIONS:
        cycles = section_cycles.get(name, [])
        out.append(_or_absent(name, [_cycle_ident(r, s, l)
                                     for r, s, l, _ in cycles]))
    return out


def _mirror_key(lfam: str, rfam: str) -> Tuple[int, int, int]:
    """Sort key seating a corner pair next to its mirror: unordered family pair
    first (so (l,r)/(r,l) collide), then ascending side before the flip."""
    i, j = _SECTION_RANK[lfam], _SECTION_RANK[rfam]
    return (min(i, j), max(i, j), 0 if i <= j else 1)


def _three_pairs() -> List[Tuple[str, str]]:
    """The distinct corner (lfam, rfam) pairs across all 3-region signatures,
    mirror-ordered. The shared enumeration order for the walk's three-region
    blocks and the middle-unknown test driver."""
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for _s, _lf, _mf, _rf in _THREE:
        if (_lf, _rf) not in seen:
            seen.add((_lf, _rf))
            pairs.append((_lf, _rf))
    pairs.sort(key=lambda p: _mirror_key(*p))
    return pairs


def _unknown_test_ops(lfam: str, rfam: str, lrule: ShiftRule, rrule: ShiftRule
                      ) -> List[Tuple[str, MultiOp, int, int]]:
    """Middle-unknown tests for one (left op, matching right op) of an (lfam,
    rfam) corner pair — the 3-region signatures whose middle is neither LOW nor
    NOT. Returns (token, evaluable MultiOp, a, c) where `a` is the SHL digit and
    `c` the rotation-encoded SHR digit shown in the token (true SHR = 8-c). The
    `a+c<=7` regime cut is deferred to individual testing."""
    lp, ls = lrule.shift_p, lrule.shift_s
    rp, rs = rrule.shift_p, rrule.shift_s
    pair = (lfam, rfam)
    if pair == ("LOW", "Identity"):          # middle ANDac; a=b from Identity
        return [(f"AND{rp}{c}", MultiOp("AND-shift", rp, None, c), rp, c)
                for c in range(1, NB)]
    if pair == ("HIGH", "ORNOT"):            # middle NOT(ANDac); ORNOTcb, a=b
        return [(f"NOTAND{rs}{rp}", MultiOp("NOTAND-shift", rs, None, rp), rs, rp)]
    if pair == ("Identity", "Identity"):     # middle XORac
        return [(f"XOR{lp}{rp}", MultiOp("XOR-shift", lp, None, rp), lp, rp)]
    if pair == ("AND", "AND"):               # middle MAJabc; b is the shared
        left, right = {lp, ls}, {rp, rs}
        shared = left & right
        b = min(shared) if shared else lp
        a = min(left - {b}) if left - {b} else lp
        c = min(right - {b}) if right - {b} else rp
        return [(f"Majority{a}{b}{c}", MultiOp("MAJ", a, b, c), a, c)]
    if pair == ("ANDNOT", "Identity"):       # middle Choice; ANDNOTba -> b,a
        b, a = lp, ls
        return [(f"Choice{a}{b}{c}", MultiOp("Ch", a, b, c), a, c)
                for c in range(1, NB)]
    if pair == ("XORNOT", "HIGH"):           # middle ORNOTba, both orientations
        # b+c=8 (Right=HIGH) so β=γ ⇒ encoded SHR idx = b; middle I_β ORNOT I_α.
        # XORNOT is symmetric (orientation unknown), so emit both — the right one
        # also lands the right regime. Returned a = 2nd token digit, c = 1st.
        return [(f"ORNOT{lp}{ls}", MultiOp("ORNOT-shift", lp, None, ls), ls, lp),
                (f"ORNOT{ls}{lp}", MultiOp("ORNOT-shift", ls, None, lp), lp, ls)]
    if pair == ("XORNOT", "ORNOT"):          # shape A (XNO) + shape B (ONX)
        c, b = rp, rs                        # ORNOTcb right; b is the matched
        a = ls if lp == b else lp            # XORNOT's other (non-matched) digit
        # Both tokens spell their operands in `shl,rot,shr` (a,b,c) order, matching
        # the Apply section (`_format_multiop_apply`): XNO = `(SHLa XORNOT ROTb) OR
        # SHRc`, ONX = `SHLa XOR OR(NOT ROTb, SHRc)`. So a middle reads the same in
        # the `Options`/`Testing` lines and where it is later applied.
        return [(f"XNO{a}{b}{c}", MultiOp("2-op", a, b, c, "A"), a, c),
                (f"ONX{a}{b}{c}", MultiOp("2-op", a, b, c, "B"), a, c)]
    return []


@dataclass
class Possibility:
    """One surviving region pick — the structured form used to build the per-bit
    apply plan. `middle` is None (1/2-region), a ShiftRule (unary filler) or a
    MultiOp (live middle). For 1-region `left` spans all 8 bits and `right` is
    None."""
    n: int
    left: ShiftRule
    right: Optional[ShiftRule]
    middle: Optional[object]
    left_len: int
    right_len: int
    middle_len: int = 0
    middle_label: str = ""        # display token for the middle region
    right_cands: List[ShiftRule] = field(default_factory=list)  # tied right ops
    ac: Optional[Tuple[Optional[int], Optional[int]]] = None  # pinned (a, c) override; None = derive from corners
    sig_key: str = ""             # SIGNATURES key; set on recipe-dispatched (2-region twin / 3-region dead) possibilities


def _poss_text(regs: Sequence[str], lens: Sequence[int]) -> str:
    """A `Options are` display line: ``<regs…> lengths <lens…>``."""
    return " " + " ".join(regs) + " lengths " + " ".join(str(n) for n in lens)


def _poss_regs_lens(pp: "Possibility", r: Optional[ShiftRule]
                    ) -> Tuple[List[str], List[int]]:
    """Region labels and their lengths for `pp` with chosen right `r` (None =
    1-region), in ``<left> [right] [middle]`` order."""
    regs = [pp.left.label] + ([r.label] if r is not None else [])
    lens = [pp.left_len] + ([pp.right_len] if r is not None else [])
    if pp.n == 3:
        regs.append(pp.middle_label)
        lens.append(pp.middle_len)
    return regs, lens


def _applied_text(poss: "Possibility") -> str:
    """The chosen possibility's ``<regs…> lengths <lens…>`` line, using the
    *applied* (post-truncation) per-region bit counts from `_assign_regions`
    rather than observed run lengths — so `Best option is` states the
    truncated result the Apply section uses."""
    regs, _obs = _poss_regs_lens(poss, poss.right)
    _plan, steps = _assign_regions(poss)
    counts = {role: len(bits) for role, _l, bits, _n in steps}
    if poss.n == 1:
        lens = [counts.get("All", NB)]
    else:
        lens = [counts.get("Left", 0)]
        if poss.right is not None:
            lens.append(counts.get("Right", 0))
        if poss.n == 3:
            lens.append(counts.get("Middle", 0))
    return _poss_text(regs, lens)


def _walk(an: Analysis) -> Tuple[
        List[str], List[Tuple[str, List[str], List["Possibility"]]],
        List[Tuple[str, str, "ShiftRule", "ShiftRule", int, int]]]:
    """Walk every signature possibility by region count, grouped into sections.
    Returns ``(preamble, sections, unknown_combos)`` where each section is ``(key,
    shift_lines, possibilities)`` and ``key`` is one of ``one``, ``two`` (all
    2-region twins, cut geometries `8-a`/`c`/`empty` merged into one block),
    ``three`` (per-filler blocks) or ``three_unknown`` (the unknown-middle preview;
    the caller folds the `_emit_testing` survivors in). ``unknown_combos`` is the
    ordered ``(lfam, rfam, left-op, right-op, left_len, right_len)`` menu the
    `three_unknown` fan-out displayed — `_emit_testing` walks exactly this, so the
    `Options`/`Testing` traversal matches the displayed order with no re-derivation.
    Each section header states its own shift and thresholds; Left/Right best are
    filtered inline (length logic family-level, shift logic operator-level — see
    `pair_block`)."""
    left_full = {name: bests for name, bests in an.section_lefts}
    right_full = {name: bests for name, bests in an.section_rights}
    ll, rl = an.left_longest, an.right_longest

    def fam_disp(fam, side):
        if fam in ("ORNOT", "ANDNOT"):
            return f"{fam} {'first' if side == 'Left' else 'second'} digit"
        return fam

    def length_kept(fam, side, threshold):
        """Best chains for `fam` on `side` meeting the length threshold (length
        logic is family-level — no shift filter here)."""
        full = left_full if side == "Left" else right_full
        return [(r, ln) for r, ln in full.get(fam, []) if ln >= threshold]

    def shift_ok(rule, side, filler_rule):
        """Whether an operator matches the section shift (digit-aware for
        ORNOT/ANDNOT). `filler_rule` None = shift any → always matches."""
        return (filler_rule is None
                or _filter_match(rule, side, filler_rule, "Right"))

    def side_hdr(fam, side, k, disp=None):
        if disp is None:
            disp = fam_disp(fam, side)
        return f"{disp} {k[0][1]}" if k else f"{disp} absent"

    def pair_block(lfam, rfam, lt, total, filler_rule, test_fn=None,
                   middle_ident=None, middle_len=None, record=True, sink=None,
                   sig_key="", combo_sink=None):
        """Family-pair header (length logic), then — only when `long` — the
        operator lines (shift logic): gate 1 = section shift (`filler_rule`; None
        = shift any) picks left ops; gate 2 = each left line lists only right ops
        whose shift matches *that* left (` none` when none, or a single ` none`
        when no left clears gate 1). `test_fn` (unknown section) appends its
        per-(left,right) tests as `so test …` and never records. `middle_ident`
        (3-region known middle) records `<left> <rights…> <middle> lengths …`;
        two-region callers (both None) record `<left> <rights…> lengths …`.
        `combo_sink` (unknown section): collect every emitted `(lfam, rfam, left-op,
        right-op, left_len, right_len)` so `_emit_testing` reads the SAME menu this
        fan-out displays, in the SAME order — no re-derivation."""
        free = (lfam, rfam) in _NO_SHARED_SHIFT_PAIRS   # corners pair shift-free
        lk = length_kept(lfam, "Left", lt)
        lk_disp = length_kept(lfam, "Left", 1)   # display: show length if nonzero
        rk = length_kept(rfam, "Right", 1)
        # Two independent length gates shown inline as `long`: left `long` ⇔ left
        # best clears threshold lt (lk non-empty); sum `long` ⇔ the two displayed
        # lengths sum >= total. Operators emit only when both fire (== is_long).
        is_long = bool(lk and rk and lk[0][1] + rk[0][1] >= total)

        # AND×AND pairs only: a left AND has BOTH digits as match keys (see
        # `_ml_match_keys`), and which right AND attaches is otherwise an unshown
        # `_filter_match`. Render one matching pass per left digit so the right
        # partner is *derived* from a visible shared-digit hit. Applies whatever
        # the middle is (MAJ / none / unknown). `free` corners ignore digits.
        two_digit = (lfam == "AND" and rfam == "AND" and not free)
        passes = ("first", "second") if two_digit else ("first",)

        def left_disp(which: str) -> str:
            return f"{lfam} {which} digit" if two_digit else fam_disp(lfam, "Left")

        def digit_match(lrule, which: str, r) -> bool:
            """Match a single left operand digit (`first`=shift_p, `second`=
            shift_s) against the right's match keys. For the one-pass case this
            reduces exactly to `_filter_match` (shift_p is the only left key)."""
            lkey = lrule.shift_p if which == "first" else lrule.shift_s
            if lkey is None:
                return True
            rkeys = _ml_match_keys(r, "Right")
            return None in rkeys or lkey in rkeys

        # Recording is DISPLAY-INDEPENDENT: one possibility per left rule, matched
        # on the full digit union (== pre-split `_filter_match`), so option
        # enumeration and selection stay byte-identical to the un-split CoT. Only
        # the display below splits by digit.
        if is_long and record and test_fn is None and sink is not None:
            for lrule in (lr for lr, _ in lk if shift_ok(lr, "Left", filler_rule)):
                rights = [r for r, _ in rk
                          if free or _filter_match(lrule, "Left", r, "Right")]
                if not rights:
                    continue
                lens = [lk[0][1], rk[0][1]]
                n_reg = 2
                if middle_ident is not None:
                    lens.append(middle_len)
                    n_reg = 3
                sink.append(Possibility(
                    n=n_reg, left=lrule, right=rights[0],
                    middle=(filler_rule if middle_ident is not None else None),
                    left_len=lk[0][1], right_len=rk[0][1],
                    middle_len=(middle_len if middle_ident is not None else 0),
                    middle_label=middle_ident or "",
                    right_cands=list(rights), sig_key=sig_key))

        out: List[str] = []
        seen: Set[Tuple[str, str]] = set()   # dedup combo across digit passes
        for which in passes:
            left_hdr = (f"{left_disp(which)} {lk_disp[0][1]}" if lk_disp
                        else f"{left_disp(which)} absent")
            if lk:
                left_hdr += " long"
            right_disp = "AND both digits" if two_digit else None
            head = f"{left_hdr} {side_hdr(rfam, 'Right', rk, right_disp)}"
            if lk_disp and rk:                   # both show a length → show the sum
                s = lk_disp[0][1] + rk[0][1]
                head += f" {s}"
                if s >= total:
                    head += " long"
            if free:
                head += " special exception ignore digits"
            out.append(head)
            if not is_long:
                continue
            lefts = [lr for lr, _ in lk if shift_ok(lr, "Left", filler_rule)]
            if not lefts:
                out.append(" none")
                continue
            for lrule in lefts:
                rights = [r for r, _ in rk
                          if free or (digit_match(lrule, which, r) if two_digit
                                      else _filter_match(lrule, "Left", r, "Right"))]
                if combo_sink is not None:
                    combo_sink.extend(
                        (lfam, rfam, lrule, r, lk[0][1], rk[0][1])
                        for r in rights
                        if (lrule.label, r.label) not in seen)
                parts = [lrule.label] + ([r.label for r in rights] or ["none"])
                if test_fn is not None and rights:
                    toks: List[str] = []
                    for r in rights:
                        toks += test_fn(lfam, rfam, lrule, r)
                    if toks:
                        parts += ["so", "test"] + toks
                out.append(f" {parts[0]}│ " + " ".join(parts[1:]))
                seen.update((lrule.label, r.label) for r in rights)
        return out

    preamble: List[str] = [
        "",
        "Left must match shift but only first digit for left ORNOT and ANDNOT count",
        "Right must match left but only second digit for right ORNOT and ANDNOT count",
        "Both must meet threshold",
        "Lowest is best"
    ]
    three_pairs = _three_pairs()
    sections: List[Tuple[str, List[str], List[Possibility]]] = []

    # ---- One region (single family spanning all 8 bits; shift any) ----
    one_lines = ["", f"Left with threshold {NB}"]
    one_poss: List[Possibility] = []
    for sig, fam in _ONE:
        m = length_kept(fam, "Left", NB) or length_kept(fam, "Right", NB)
        if not m:
            one_lines.append(f"{fam_disp(fam, 'Left')} absent")
            continue
        one_lines.append(f"{fam_disp(fam, 'Left')} {m[0][1]} long")
        for rule, _ in m:
            one_lines.append(f" {rule.label}")
            one_poss.append(Possibility(
                n=1, left=rule,
                right=None, middle=None, left_len=NB, right_len=0))
    sections.append(("one", one_lines, one_poss))

    # ---- Two regions: a single block with one `Left right` header and one
    # `Options are`. Twins are walked grouped by their length-candidate SHAPE
    # (`_TWO_BUCKETS`: `c,8-c` / `8-a,a` / two-candidate / HIGH·ORNOT) rather than by
    # cut, so same-shape `pass true lengths` lines render adjacently. Selection is
    # order-robust, so the order is display-only. Ambiguous twins (`ANDNOTba·Ib`)
    # need no special-casing: `_recover_ac` returns both cuts and the length-diff
    # picks the closer.
    # Only tested when the longest corner runs can jointly tile all NB bits
    # (`ll + rl >= NB`); otherwise any two-region cut leaves a gap, so the block
    # is skipped entirely (matches the no-middle gate narrated in the cycle
    # survey above).
    if ll + rl >= NB:
        lt2 = max(ll - 1, 1)
        two_header = (f"Left right no middle with left threshold "
                      f"max({ll}-1,1)={lt2} and total threshold {NB}")
        for _bi, (_label, _keys) in enumerate(_TWO_BUCKETS):
            two_lines = [two_header] if _bi == 0 else []
            two_poss: List[Possibility] = []
            for _sig in _keys:
                _lfam, _rfam = SIG_BY_KEY[_sig].fams
                two_lines += pair_block(_lfam, _rfam, lt2, NB, None, sink=two_poss,
                                        sig_key=_sig)
            sections.append(("two", two_lines, two_poss))

    # ---- Three regions: per-filler blocks then the unknown-middle preview.
    # Left threshold == the filler's start bit; total threshold (NB-1)-end+start
    # == NB-length (bits the corners must jointly cover). The actual unknown-
    # middle testing (`_emit_testing`) and its survivors are folded into this
    # section by the caller. ----
    three_lines: List[str] = []
    three_poss: List[Possibility] = []
    sig3 = {(lf, mf, rf): s for s, lf, mf, rf in _THREE}  # 3-region key by family triple
    fillers = _cycle_survey(an.section_cycles, _middle_set(ll, rl))
    for name, rule, start, length in fillers:
        # Pairs whose 3-region grammar middle IS this filler's family are the
        # only ones that record a possibility; the rest are shown for display.
        valid = {(lf, rf) for _s, lf, mf, rf in _THREE if mf == name}
        end = start + length - 1
        total = NB - length
        shift = "any" if rule.shift_p is None else str(rule.shift_p)
        three_lines.append("")
        with_shift = "with" if shift == "any" else f"with SHIFT {shift} and"
        three_lines.append(f"Left right middle {_cycle_ident(rule, start, length)} "
                           f"{with_shift} left threshold {start} and "
                           f"total threshold {NB - 1}-{end}+{start}={total}")
        for lfam, rfam in three_pairs:
            is_valid = (lfam, rfam) in valid
            three_lines += pair_block(
                lfam, rfam, start, total, rule,
                middle_ident=_cycle_ident(rule, start, length) if is_valid else None,
                middle_len=length if is_valid else None,
                record=is_valid, sink=three_poss,
                sig_key=sig3.get((lfam, name, rfam), ""))
    sections.append(("three", three_lines, three_poss))

    # The unknown (live) middle is its own section: just the matching corner pairs
    # here (no per-pair recording). `_render_and_select` folds in its
    # `Options are` block (every `a+c<=7` candidate's regime check + computed
    # left/right/middle) BEFORE the brute-force `_emit_testing` that verifies the
    # middle bits.
    unknown_lines = ["Left right middle unknown with left threshold 1 and "
                     "total threshold 2 and pass if overlap negative"]
    unknown_combos: List[Tuple[str, str, ShiftRule, ShiftRule, int, int]] = []
    for lfam, rfam in three_pairs:
        unknown_lines += pair_block(lfam, rfam, 1, 2, None, record=False,
                                    combo_sink=unknown_combos)
    sections.append(("three_unknown", unknown_lines, []))

    return preamble, sections, unknown_combos


def _emit_gap_eval(lines: List[str], token: str, mop: MultiOp,
                   gap: Sequence[int], cols: Columns) -> bool:
    """One Apply-format line per gap bit for a test `token`/`mop`:

        {bit}│{token} = {fam}({IN operands}) = {fam}({operand cols}) = {got}
                       {per-4-bit-group match/not}

    Columns and result use 4-bit grouping; the per-group verdicts truncate at
    the first failing group. Returns True iff every gap bit matches output."""
    n = cols.n_examples
    inp, outp = cols.input_columns, cols.output_columns
    in_or_0 = lambda s, bit: "0" if s is None else f"IN{(s + bit) % NB}"
    for bit in gap:
        lc = _atom_column_wrap(mop.shl, bit, inp, n)
        tc = _atom_column_wrap(mop.rot, bit, inp, n)
        rc = _atom_column_wrap(mop.shr, bit, inp, n)
        # Shared reduction: identical grammar to the Apply renderer, lifted to
        # whole-column vectors (the 2-op shapes expand instead of the opaque
        # XNO(a,b,c) form the model couldn't reproduce in one jump).
        rhs = _multiop_rhs(mop.family, mop.shape,
                           in_or_0(mop.shl, bit), in_or_0(mop.rot, bit),
                           in_or_0(mop.shr, bit), lc, tc, rc)
        got = _multiop_eval_column(mop, bit, inp, n)
        expected = outp[bit]
        groups = ["match" if got[i:i + 4] == expected[i:i + 4] else "not"
                  for i in range(0, n, 4)]
        first_not = next((i for i, g in enumerate(groups) if g == "not"), None)
        per_group = " ".join(groups[:first_not + 1]
                             if first_not is not None else groups)
        lines.append(f"{bit}│{token} = {rhs} {per_group}")
        if first_not is not None:
            return False
    return True


def _emit_testing(an: Analysis,
                  combos: Sequence[Tuple[str, str, "ShiftRule", "ShiftRule",
                                         int, int]]
                  ) -> Tuple[List["Possibility"],
                             Dict[int, Tuple[bool, List[str]]]]:
    """Build every unknown candidate, walking `combos` — the SAME `(lfam, rfam,
    left-op, right-op, left_len, right_len)` menu the `three_unknown` fan-out
    displayed, in the SAME order (`_walk` builds it; no re-derivation here) — and
    pre-run the gap-test for each LIVE (`a+c <= 7`) one. Returns `(candidates,
    gap_by_id)`: `candidates` is the full sweep (listed in `Options are`, in menu
    order); `gap_by_id` maps `id(candidate)` -> `(passed, Apply-format lines)` for
    the live ones (`passed` ⇒ it reproduced every gap bit, i.e. eligible to win). The
    renderer shows a `Testing` block + a `test` mark only for the candidates it must
    verify (L1 diff < running min); surviving is what the test decides, so it can't
    be read off the length sweep alone."""
    candidates: List[Possibility] = []
    gap_by_id: Dict[int, Tuple[bool, List[str]]] = {}

    for lfam, rfam, lrule, rrule, llen, rlen in combos:
        for token, mop, a, shr_digit in _unknown_test_ops(
                lfam, rfam, lrule, rrule):
            c = _denorm_shr(shr_digit)       # true SHR; live iff overlap negative
            le, rs = _ac_cut(a, c)
            gap = list(range(le + 1, rs))
            # Pin (a, c) so the candidate's regime band/lengths match the tested
            # orientation (ORNOT-shift stores shl/shr swapped vs its a/c; `_ac_cut`
            # is symmetric so boundaries are unchanged). Every token is a candidate
            # for the `Options` sweep; only a live (a+c<=7) middle is gap-tested.
            cand = Possibility(
                n=3, left=lrule, right=rrule, middle=mop,
                left_len=llen, right_len=rlen, middle_len=len(gap),
                middle_label=token, ac=(a, c))
            candidates.append(cand)
            if a + c <= 7 and gap:           # live middle — pre-run its gap check
                gl: List[str] = []
                passed = _emit_gap_eval(gl, token, mop, gap, an.cols)
                gap_by_id[id(cand)] = (passed, gl)
    return candidates, gap_by_id


def explain(examples: Sequence[Tuple[str, str]], query: str = "") -> Optional[str]:
    """The full structured trace through the `Best option is` line."""
    an = analyze(examples)
    _chosen, parts = _render_and_select(an)
    return "\n".join(list(an.trace_lines) + [""] + parts)


# Given a list of possibilities, filter by shifts and choose based on the first
# best expected and observed length match.


def _denorm_shr(shift: Optional[int]) -> Optional[int]:
    """A SHR atom's parameter `c` from its stored (normalized rotation) shift:
    `SHRc` reads `I_{i-c}`, i.e. normalized shift `(8 - c) % 8`, so `c = (8 -
    shift) % 8`. None passes through (atom absent)."""
    return None if shift is None else (8 - shift) % 8


def _rot_operand(rule: ShiftRule) -> Optional[int]:
    """The ROT (β) operand shift of an *asymmetric* binary region, by position —
    breaks the `a+c=8` collapse tie. `ORNOT` (`I_γ ORNOT I_β`) carries ROT in its
    second operand; `ANDNOT` (`I_β ANDNOT I_α`) in its first. Symmetric ops (incl.
    `XORNOT`) have no fixed ROT position → None."""
    if rule.family == "ORNOT":
        return rule.shift_s
    if rule.family == "ANDNOT":
        return rule.shift_p
    return None


def _shared_b(L: ShiftRule, R: ShiftRule) -> List[Optional[int]]:
    """The ROT `b` candidate(s): the shift shared by both corners. On an `a+c=8`
    collision (both corners share two shifts) an asymmetric corner fixes ROT by
    position; two symmetric corners (`ANDab·ANDbc`) can't, so both readings are
    returned for the length diff to decide. Otherwise a single candidate (min of
    the shared set, or None)."""
    ls = [s for s in (L.shift_p, L.shift_s) if s is not None]
    rs = [s for s in (R.shift_p, R.shift_s) if s is not None]
    shared = set(ls) & set(rs)
    if len(shared) == 2:
        rot = _rot_operand(R) or _rot_operand(L)
        return [rot] if rot is not None else sorted(shared)
    return [min(shared) if shared else None]


def _side(shifts: List[int], b: Optional[int]) -> Optional[int]:
    """The corner's own (non-ROT) atom: its shift other than `b`; None for a pure
    ROT corner (no leftover shift)."""
    rest = [s for s in shifts if s != b]
    return rest[0] if rest else None


def _recover_ac(poss: Possibility) -> List[Tuple[Optional[int], Optional[int]]]:
    """The candidate shift-cuts `(a, c)` that fix the region boundaries — `a` =
    SHL shift (used directly), `c` = SHR shift (`_denorm_shr`); either may be None
    when that atom is absent. Returns a LIST: empty = no cut placeable; usually
    one; an ambiguous twin returns two for the length-diff selection. Fast-paths:
    pinned `poss.ac`, 1-region (none), live-form 3-region middle. Otherwise the
    signature's `recipe` (keyed by `poss.sig_key`) reads `(a, c)` off the corners:
      const            — 2-region const-corner twin; `sig.cut` picks SHL `8-a` vs
                         SHR `c`; a binary corner offers both atom readings.
      both_single      — both corners single-shift (`Ia·LOW·Ic` = SHLa XOR SHRc).
      both_binary      — both binary; an `a+c=8` collision yields two `b` readings.
      dead_high_ornot  — `HIGH·NOTb·ORNOTcb`: b from the middle NOT, c from R.
      dead_xornot_high — `XORNOTab·NOT·HIGH`: single cut `(m, 8-other)`.
      shl_1op/shr_1op  — genuine SHL-/SHR-only 1-op; the non-ROT atom is a/c.
      collapse_a_eq_b  — collapsed MAJ `a=b` (`Ia·ANDbc`).
      collapse_b_eq_c  — collapsed MAJ `b+c=8` (`ANDab·Ib`).
      andnot_ib        — ambiguous `ANDNOTba·Ib`: two competing cut sites."""
    if poss.ac is not None:                    # caller pinned the cut (selector winner)
        return [poss.ac]
    if poss.right is None:                     # 1-region
        return []
    if isinstance(poss.middle, MultiOp):       # 3-region, live-form middle
        return [(poss.middle.shl, _denorm_shr(poss.middle.shr))]
    sig = SIG_BY_KEY.get(poss.sig_key)
    if sig is None:
        return []
    L, R = poss.left, poss.right
    ls = [s for s in (L.shift_p, L.shift_s) if s is not None]
    rs = [s for s in (R.shift_p, R.shift_s) if s is not None]
    r = sig.recipe
    if r == "const":                           # live corner carries the shift(s)
        atoms = ls or rs
        if sig.cut == _CUT_8A:
            return [(d, None) for d in atoms]
        return [(None, _denorm_shr(d)) for d in atoms]
    if r == "both_single":
        return [(ls[0], _denorm_shr(rs[0]))]
    if r == "both_binary":                     # collision -> two `b` readings
        return [(_side(ls, b), _denorm_shr(_side(rs, b))) for b in _shared_b(L, R)]
    if r == "dead_high_ornot":                 # b from middle NOT, c from R's non-ROT
        return [(poss.middle.shift_p, _denorm_shr(R.shift_p))]
    if r == "dead_xornot_high":                # single cut (m, 8-other)
        m_ = poss.middle.shift_p
        other = next((s for s in (L.shift_p, L.shift_s) if s != m_), m_)
        return [(m_, NB - other)]
    b = _shared_b(L, R)[0]                      # ROT shared by both corners
    if r == "shl_1op":
        return [(_side(ls, b), None)]
    if r == "shr_1op":
        return [(None, _denorm_shr(_side(rs, b)))]
    if r == "collapse_a_eq_b":
        return [(L.shift_p, _denorm_shr(_side(rs, b)))]
    if r == "collapse_b_eq_c":
        return [(_side(ls, b), _denorm_shr(b))]
    if r == "andnot_ib":
        a = _side(ls, b)
        return [(a, None), (a, _denorm_shr(b))]
    return []                                  # oneregion/multiop: fast-paths handle


def _ac_cut(a: Optional[int], c: Optional[int]) -> Tuple[int, int]:
    """``(left_end, right_start)`` from recovered ``(a, c)``: Left ends
    ``min(7-a, c-1)``, Right starts ``max(c, 8-a)``. An absent atom is filled
    with ``8-other`` (the empty-middle ``a+c=8`` point) so a lone 1-op shift's
    single cut uses the same formula."""
    if a is None:
        a = 8 - c
    if c is None:
        c = 8 - a
    return min(7 - a, c - 1), max(c, 8 - a)


def _assign_regions(
    poss: Optional[Possibility],
) -> Tuple[List[Optional[object]], List[Step]]:
    """Lay the regions onto the 8 bits from the recovered shift magnitudes
    (`_recover_ac`): Left ends `min(7-a, c-1)`, Right starts `max(c, 8-a)`, Middle
    the gap (`_ac_cut`). An unrecoverable cut yields the empty (all-zero) plan
    defensively. Returns (plan, steps), steps = (role, label, bits, note)."""
    plan: List[Optional[object]] = [None] * NB
    if poss is None:
        return plan, []
    if poss.n == 1:
        return [poss.left] * NB, [("All", poss.left.label, list(range(NB)), "")]
    acs = _recover_ac(poss)
    ac = acs[0] if acs else None
    if ac is None:                             # unrecoverable (dropped) -> all-zero
        return plan, []
    left_end, right_start = _ac_cut(*ac)
    mid_lo, mid_hi = left_end + 1, right_start
    left_hi = left_end + 1
    if poss.n == 2:                            # single cut, no middle region
        # cut at the clean corner's boundary: left_end+1 normally, but right_start
        # when the left absorbed the merged middle (`Ia·ANDbc`, a=b — left is the
        # single-shift Identity, right the binary corner).
        cut = (right_start if poss.left.shift_s is None
               and poss.right is not None and poss.right.shift_s is not None
               else left_end + 1)
        mid_lo = mid_hi = left_hi = cut
    steps: List[Step] = []

    def place(role: str, label: str, region: object, rng: range) -> None:
        bits = [b for b in rng if 0 <= b < NB]
        for b in bits:
            plan[b] = region
        steps.append((role, label, bits, ""))

    place("Left", poss.left.label, poss.left, range(0, left_hi))
    if mid_hi > mid_lo and poss.middle is not None:
        place("Middle", poss.middle_label or "middle", poss.middle,
              range(mid_lo, mid_hi))
    if poss.right is not None:
        place("Right", poss.right.label, poss.right, range(mid_hi, NB))
    return plan, steps


def _regime_band(a: int, c: int) -> str:
    """The signed region overlap ``a-v=a+c-8`` in the VISIBLE SHR/ROT digit
    ``v=8-c`` (`NB-c`), not the hidden true ``c``. Its SIGN is the regime
    (negative=live gap / zero=empty tile / positive=dead overlap) and its
    MAGNITUDE is the middle length. Informational only — never rejects."""
    return f"{a}-{NB - c}={a + c - NB}"


def _regime_check(poss: "Possibility") -> str:
    """The regime band to SHOW (informational — never rejects; selection is pure
    L1 length diff). `_regime_band` for any 3-region or a gated signature (the 4
    collapse 2-region twins) with a recoverable ``(a, c)``; "" otherwise
    (1-region, ungated twins, unrecoverable cut)."""
    if poss.right is None or poss.n == 1:
        return ""
    sig = SIG_BY_KEY.get(poss.sig_key)
    show = poss.n == 3 or (sig is not None and sig.gate is not None)
    if not show:
        return ""
    acs = _recover_ac(poss)
    if not acs or acs[0][0] is None or acs[0][1] is None:
        return ""
    return _regime_band(*acs[0])


def _gate_sat(gate: Optional[str], s: int) -> Optional[bool]:
    """Whether sum ``s = a+c`` satisfies the regime ``gate`` — the sign of the
    overlap ``s-8`` (``negative`` / ``zero`` / ``positive``); None when no gate."""
    if gate is None:
        return None
    return {"negative": s < NB, "zero": s == NB, "positive": s > NB}[gate]


def _resolved_gate(poss: "Possibility", a: int, c: int) -> Optional[str]:
    """The regime gate governing ``poss`` at cut ``(a, c)``. A 3-region family
    triple that maps to BOTH a live (`a+c<=7`) and a dead (`a+c>=9`) variant is
    resolved by the actual ``a+c`` — fixing the table-order collision that let one
    variant shadow the other. Single-signature triples and 2-region twins return
    their own gate."""
    sig = SIG_BY_KEY.get(poss.sig_key)
    if sig is None:
        return None
    s = a + c
    if len(sig.fams) == 3:
        for key in _SIG3_BY_TRIPLE.get(sig.fams, ()):
            g = SIG_BY_KEY[key].gate
            if _gate_sat(g, s):
                return g
    return sig.gate


# The regime legend (shown once, up top). Each per-section `Options are` block
# annotates its candidates against these regimes by the sign of the overlap
# o=a+c-8 (see `_regime_check`). NOTE: these are SHOWN for context, not enforced —
# selection is pure L1 length diff and never rejects on the regime.
_CHECKS_LEGEND = [
    "Checks",
    " AND AND zero",
    " XORNOT ORNOT zero",
    " Identity AND negative",
    " AND Identity negative",
    " AND AND LOW positive",
    " Identity AND LOW positive",
    " AND Identity LOW positive",
    " Identity Identity LOW positive",
    " XORNOT ORNOT NOT positive",
    " XORNOT HIGH NOT positive",
    " HIGH ORNOT NOT positive",
]


def _strip_blanks(lines: Sequence[str]) -> List[str]:
    """Drop leading/trailing empty strings (interior blanks kept)."""
    lo, hi = 0, len(lines)
    while lo < hi and lines[lo] == "":
        lo += 1
    while hi > lo and lines[hi - 1] == "":
        hi -= 1
    return list(lines[lo:hi])


def _corner(p: str, q: str) -> str:
    """The corner-reach ``min(p,q)``, collapsed to a single operand when the two
    sides agree. They coincide exactly when one shift atom is absent and is filled
    from the other (e.g. SHR-only 2-region: ``a:=8-c`` makes both reaches ``8-c``),
    so the ``min`` only carries information when both atoms are present (3-region,
    or the `Ia·ANDbc`/`ANDab·Ib` collapse twins)."""
    return p if p == q else f"min({p},{q})"


def _shift_lengths(pp: Possibility, r: Optional[ShiftRule],
                   ac: Optional[Tuple[Optional[int], Optional[int]]]
                   ) -> Tuple[int, int, str, str]:
    """``(left, right, left_expr, right_expr)`` for one candidate cut ``ac`` of
    ``(pp, r)``. Integer lengths are exact; *expr* strings use the VISIBLE SHR/ROT
    digit ``v=8-c``: left ``min(8-a,c)`` shows ``min(8-a,8-v)``, a 3-region right
    ``min(a,8-c)`` shows ``min(a,v)``, a twin right is ``8-left``. A `min` whose
    two sides agree collapses to one operand (`_corner`). 1-region is ``(8, 0, …)``;
    ``ac`` None returns the observed lengths as bare numbers."""
    if pp.n == 1 or r is None:
        return NB, 0, str(NB), ""
    if ac is None:                            # no shift -> observed lengths are exact
        return pp.left_len, pp.right_len, str(pp.left_len), str(pp.right_len)
    a, c = ac
    af = a if a is not None else NB - c        # fill the absent atom (see _ac_cut)
    cf = c if c is not None else NB - af
    vf = NB - cf                               # VISIBLE SHR/ROT digit (8-c); display in v
    left_end, right_start = _ac_cut(a, c)
    left = left_end + 1
    left_expr = _corner(f"8-{af}", f"8-{vf}")  # c = 8-v, so min(8-a,c) reads min(8-a,8-v)
    if pp.n == 2:                             # single cut: corners tile, no middle
        if pp.left.shift_s is None and r.shift_s is not None:
            # left absorbed the merged middle (`Ia·ANDbc`, a=b): it reaches
            # right_start, not 8-a (cut at the binary RIGHT corner).
            left = right_start
        # Collapse each 2-region corner to its single regime-winning boundary
        # operand (the twin always tiles and is gated, so the winner is
        # determinate); right is the tile complement `8-left`. No `min` survives
        # in 2-region — it carries information only in the 3-region corners.
        lwin = f"8-{af}" if (NB - af) == left else f"8-{vf}"
        return left, NB - left, lwin, f"8-{left}"
    return left, NB - right_start, left_expr, _corner(str(af), str(vf))


def _score_possibility(pp: Possibility, r: Optional[ShiftRule], show_gate: bool = True
                       ) -> List[Tuple[str, int,
                                       Optional[Tuple[Optional[int], Optional[int]]]]]:
    """One ``(segment, diff, ac)`` per shift-derived cut of ``(pp, r)`` (an
    ambiguous twin yields two, rendered as adjacent ``│…`` segments). Each segment
    is ``│[needs {gate} ]{a-v=o} pass true lengths {Lexpr} {Rexpr} [M]
    |L-obsL|+|R-obsR|=diff`` in the VISIBLE digit ``v=8-c`` — ``o=a+c-8`` is the
    signed overlap (sign = regime, |o| = middle length). The band drops for an
    ungated row; `needs {gate}` also drops when `show_gate` is False (unknown
    section, gate stated in its header). ``diff`` ignores the middle and is 0 for
    1-region. A cut whose ``a+c`` violates its (collision-resolved) regime gate is
    dropped, as is a 2/3-region with no shift-recoverable cut — yielding **no**
    segments removes the candidate from the contest. Lowest diff wins."""
    band = "" if r is None else _regime_check(replace(pp, right=r))
    if pp.n == 1 or r is None:                # 1-region: spans 8 bits, no cut to recover
        diff = abs(NB - pp.left_len)
        return [(f"│ pass true lengths {NB} |{NB}-{pp.left_len}|={diff}", diff, None)]
    out = []
    acs = _recover_ac(replace(pp, right=r))   # empty -> no segments -> candidate dropped
    for ac in acs:
        if ac and None not in ac:             # gate filter: the cut must match its regime
            fail_gate = _resolved_gate(pp, ac[0], ac[1])
            if _gate_sat(fail_gate, ac[0] + ac[1]) is False:
                # Don't silently drop a gate-failing cut: show it as a band-only
                # rejection (diff None ⇒ ineligible to win) so the reader sees the
                # recovered overlap and the regime it needed — the same `needs
                # <gate> {band}` grammar the surviving lines use. Parallel to the
                # dead-middle branch below; only `_recover_ac`-empty drops stay silent.
                need = f"needs {fail_gate} " if fail_gate and show_gate else ""
                out.append((f"│ {need}{_regime_band(*ac)}", None, ac))
                continue
        # Unknown live-middle sweep: a dead/empty regime (overlap a+c-8 >= 0) is no
        # live middle, so show just its regime band (the non-negative overlap is the
        # rejection, per the header) — no length score (diff None ⇒ ineligible),
        # parallel to its `Testing` regime line. Keeps the sweep complete in order.
        if (isinstance(pp.middle, MultiOp) and pp.ac is not None
                and ac and None not in ac and ac[0] + ac[1] >= NB):
            out.append((f"│{_regime_band(*ac)}", None, ac))
            continue
        # An ambiguous form (two cuts) carries a per-segment band so each regime
        # calc matches its own lengths (e.g. `ANDab·ANDbc`).
        seg_band = (_regime_band(*ac) if len(acs) > 1 and band and ac and None not in ac
                    else band)
        # `needs <gate>` names the regime this gated form must land in (3-region
        # always gated; the 4 collapse/empty 2-region twins). Shown before the band
        # so the requirement reads ahead of the value that meets it.
        gate = (_resolved_gate(pp, ac[0], ac[1])
                if seg_band and ac and None not in ac else None)
        if gate is None and seg_band and pp.n == 3:
            gate = "negative"                    # synthesized live middle (no sig_key)
        left, right, lexpr, rexpr = _shift_lengths(pp, r, ac)
        diff = abs(left - pp.left_len) + abs(right - pp.right_len)
        ld = lexpr if lexpr == str(left) else f"{lexpr}={left}"
        rd = rexpr if rexpr == str(right) else f"{rexpr}={right}"
        lengths = f"true lengths {ld} {rd}"
        if pp.n == 3:
            mid = NB - left - right           # middle = remainder of the tile
            lengths += f" {mid}"
        body = (f"{lengths} "
                f"|{left}-{pp.left_len}|+|{right}-{pp.right_len}|={diff}")
        # No space after `│` when the band (a number) follows; keep it before the
        # `needs` word and before a bare `pass`.
        need = f"needs {gate} " if gate and show_gate else ""
        head = (f"│ {need}{seg_band} pass " if need
                else f"│{seg_band} pass " if seg_band else "│ pass ")
        out.append((head + body, diff, ac))
    return out


def _shorter_chain_cands(an: Analysis,
                         sections: Sequence[Tuple[str, List[str],
                                                  List[Possibility]]]
                         ) -> List[Possibility]:
    """Rebuild every recorded 2/3-region possibility on the SUB-MAXIMAL corner runs
    (those `_tied_longest` dropped) — both sides crossed. The no-winner fallback
    renders these through the normal `render_cands` so the gate + L1 pick a shorter,
    gate-consistent corner when the greedy longest run over-ran a region boundary.
    Dedup by (sig_key, left label, right label)."""
    out: List[Possibility] = []
    seen: Set[Tuple] = set()
    for _name, _lines, cands in sections:
        for pp in cands:
            if pp.right is None:
                continue
            l_runs = (_all_corner_runs(an, pp.left.family, 0)
                      or [(pp.left, 0, pp.left_len, None)])
            r_runs = (_all_corner_runs(an, pp.right.family, NB - 1)
                      or [(pp.right, NB - pp.right_len, pp.right_len, None)])
            for lr, _ls, llen, _lf in l_runs:
                for rr, _rs, rlen, _rf in r_runs:
                    key = (pp.sig_key, lr.label, rr.label)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(replace(pp, left=lr, right=rr,
                                       left_len=llen, right_len=rlen,
                                       right_cands=[rr]))
    return out


def _example_fit(cand: "Possibility", cols: Columns) -> int:
    """Output bits the candidate's 3-region rule MISPREDICTS across the examples
    (proper SHL/SHR zeroing, via its own apply plan). 0 = reproduces every example
    exactly; a small count = a clean rule the noise nicked in a few middle bits."""
    plan = _assign_regions(cand)[0]
    inp, outp, n = cols.input_columns, cols.output_columns, cols.n_examples
    miss = 0
    for j in range(n):
        q = "".join(inp[bit][j] for bit in range(NB))
        got = _compute_answer(q, plan)
        miss += sum(g != outp[bit][j] for bit, g in enumerate(got))
    return miss


def _best_fit_guess(an: Analysis, candidates: Sequence["Possibility"]
                    ) -> Optional["Possibility"]:
    """The closest grammar rule when none reproduces the examples exactly (noised
    rows): among the longest-corner unknown candidates, the one reproducing the MOST
    example bits, ties to the best length-fit. Recovers the underlying (almost always
    2-op) signature that a few flipped middle bits hid — predicts ~39/53 of the
    train fallbacks vs ~13/53 for the per-bit HIGH-default survey. None if the
    section produced no candidate to fall back on."""
    if not candidates:
        return None

    def key(c):
        segs = _score_possibility(c, c.right, False)
        l1 = min((d for _, d, _ in segs if d is not None), default=NB * NB)
        return (_example_fit(c, an.cols), l1)

    return min(candidates, key=key)


def _emit_shape_count(lines: List[str], token: str, mop: MultiOp,
                      gap: Sequence[int], cols: Columns,
                      show_tally: bool = True) -> int:
    """Per-middle-bit match tally for `token`/`mop` — the noise-tolerant cousin of
    `_emit_gap_eval` (no exact-match short-circuit). Each line shows the predicted
    column, its per-4-example `match`/`not` verdicts, and (when `show_tally`) the
    running count of matched groups as ``prev+this=total`` (bare on the first line).
    `show_tally` is off when the shape isn't being contested (the GTE-direct path,
    where the verdicts just confirm the exact fit). Returns the total matched-group
    count; the higher count wins the XNO/ONX/GTE (Shape A/B/C) pick."""
    n = cols.n_examples
    inp, outp = cols.input_columns, cols.output_columns
    in_or_0 = lambda s, bit: "0" if s is None else f"IN{(s + bit) % NB}"
    run = 0
    for bit in gap:
        lc = _atom_column_wrap(mop.shl, bit, inp, n)
        tc = _atom_column_wrap(mop.rot, bit, inp, n)
        rc = _atom_column_wrap(mop.shr, bit, inp, n)
        # Same interpretable reduction as Apply/Testing — the noised-row fallback
        # is where the XNO/ONX shape contest is decided, so it must expand too.
        rhs = _multiop_rhs(mop.family, mop.shape,
                           in_or_0(mop.shl, bit), in_or_0(mop.rot, bit),
                           in_or_0(mop.shr, bit), lc, tc, rc)
        got = _multiop_eval_column(mop, bit, inp, n)
        groups = ["match" if got[i:i + 4] == outp[bit][i:i + 4] else "not"
                  for i in range(0, n, 4)]
        m = groups.count("match")
        tally = f" {run}+{m}={run + m}" if show_tally else ""   # first line reads 0+m=m
        run += m
        lines.append(f"{bit}│{token} = {rhs} {' '.join(groups)}{tally}")
    return run


def _bestfit_block(an: Analysis, candidates: Sequence["Possibility"],
                   combined: bool = False, cur: str = ""
                   ) -> Optional[Tuple["Possibility", List[str]]]:
    """The no-winner fallback: take the longest-corner XORNOT/ORNOT pair (via
    `_best_fit_guess`), then DECIDE the unknown middle. Shape C `GTE` (Shape A with its
    all-ones cell cleared) is tried first: when it reproduces EVERY example exactly we
    commit straight to it — no contest, the per-bit verdicts just confirm the fit. Only
    when it doesn't fit do we fall back to running Shape A `XNO` vs Shape B `ONX` over
    the middle bits and counting matched example groups (`_emit_shape_count`); the higher
    count wins, A taking ties (`>=`). Returns `(chosen, lines)`: the recovered rule and
    its trace block. Non-2-op or empty-middle picks (no shape contest to run) name the
    middle directly. None when there is no candidate at all.

    `combined` renders BODY-ONLY for the unified `_render_and_select` fallback: the
    leading `Best option is none so try the longest left … pair` prefix and the
    trailing `Best option is …` winner line are both dropped (the caller supplies the
    `so try others …` families header and the final `Best option is`)."""
    g = _best_fit_guess(an, candidates)
    if g is None:
        return None
    L, R = g.left, g.right
    sibs: Dict[str, "Possibility"] = {}
    gap: List[int] = []
    if g.ac and isinstance(g.middle, MultiOp) and g.middle.family == "2-op":
        for cand in candidates:
            if (isinstance(cand.middle, MultiOp) and cand.middle.family == "2-op"
                    and cand.ac == g.ac and cand.middle.rot == g.middle.rot):
                sibs[cand.middle_label[:3]] = cand
        le, rs = _ac_cut(*g.ac)
        gap = list(range(le + 1, rs))
    if not ({"XNO", "ONX"} <= set(sibs)) or not gap:   # no A/B contest to run
        if combined:
            return g, []                     # caller emits the final `Best option is`
        return g, [f"Best option is none so try the longest left {L.family} "
                   f"right {R.family} pair{_applied_text(g)}"]
    # Corner lengths are the principled `(a, c)` cut (the same `true lengths min(…)`
    # form the `Options` lines use), NOT the observed run lengths — noise can over-
    # extend a run past the cut, so the cut is what we actually apply.
    left, right, lexpr, rexpr = _shift_lengths(g, g.right, g.ac)
    ld = lexpr if lexpr == str(left) else f"{lexpr}={left}"
    rd = rexpr if rexpr == str(right) else f"{rexpr}={right}"
    # Shape C (GTE) reuses Shape A's corners and (a, b, c); only its middle combiner
    # differs. If it reproduces every example exactly it is THE rule — go straight to
    # it, announcing the GTE middle and its length up front, then verify per-bit.
    xno = sibs["XNO"].middle
    gte = replace(sibs["XNO"],
                  middle=MultiOp("2-op", xno.shl, xno.rot, xno.shr, "C"),
                  middle_label=f"GTE{xno.shl}{xno.rot}{xno.shr}")
    if _example_fit(gte, an.cols) == 0:
        md = f"{NB}-{left}-{right}={NB - left - right}"
        gte_head = (f"Best option is {cur} so try GTE middle "
                    f"{L.label} {R.label} "
                    f"{gte.middle_label} true lengths {ld} {rd} {md}") if combined else (
                    f"Best option is none so try the longest left {L.family} right "
                    f"{R.family} pair with GTE middle {L.label} {R.label} "
                    f"{gte.middle_label} true lengths {ld} {rd} {md}")
        lines = [gte_head, f"Testing {gte.middle_label}"]
        _emit_shape_count(lines, gte.middle_label, gte.middle, gap, an.cols,
                          show_tally=False)
        if not combined:
            lines.append(f"Best option is{_applied_text(gte)}")
        return gte, lines
    # Shape C didn't fit, so it can't win — run the original Shape A vs B contest.
    head = (f"Best option is still currently {cur} so try GTE middle "
            f"{L.label} {R.label} true lengths {ld} {rd}") if combined else (
            f"Best option is none so try the longest left {L.family} "
            f"right {R.family} pair {L.label} {R.label} true lengths {ld} {rd}")
    lines = [head]
    totals: Dict[str, int] = {}
    for tag in ("XNO", "ONX"):
        cand = sibs[tag]
        lines.append(f"Testing {cand.middle_label}")
        totals[tag] = _emit_shape_count(lines, cand.middle_label, cand.middle,
                                        gap, an.cols)
        lines.append(f"{cand.middle_label} matched {totals[tag]}")
    win = sibs["XNO"] if totals["XNO"] >= totals["ONX"] else sibs["ONX"]
    lines.append(f"So the first one that matches more is {win.middle_label}")
    if not combined:
        lines.append(f"Best option is{_applied_text(win)}")
    return win, lines


# A region-walk winner whose corner length-diff (absolute sum) reaches this is
# rejected in favour of the unified fallback — a badly-overshot corner that still
# won the gate (e.g. 649bebaf's sum-11 pick) is worse than retrying shorter corners.
_FALLBACK_SUM_GATE = 4


def _render_and_select(an: Analysis, *, trace: Optional[dict] = None
                       ) -> Tuple[Optional[Possibility], List[str]]:
    """Build the full per-section trace and choose the global pick. Each shift-check
    section is followed by its `Options are` block (regime check then length
    diff per candidate; ``best`` marks each new running-minimum diff), ending on
    `Best option is …`. The pick is the smallest diff, earliest in section
    order. If none passes the gate, retry the SHORTER chain (`_shorter_chain_cands`,
    same gate + L1 rendering); failing that, the first possibility by section order.
    Returns (chosen, parts). When `trace` is given, records which fallback block
    fired in `trace["fallback_block"]`: 0=gate winner kept, 1=spurious over-run
    (shorter corner), 2=GTE/Shape-A-B bestfit, 3=per-bit fill (see `classify_fallback`)."""
    preamble, sections, unknown_combos = _walk(an)
    # Build the full candidate sweep and pre-run each live one's gap-test;
    # `render_unknown` verifies/displays only the lazy subset (diff < running min).
    # `survivors` win the pick. The section walk is ALWAYS emitted (even when nothing
    # wins) — the fallback survey is appended after it, never in place of it.
    candidates, gap_by_id = _emit_testing(an, unknown_combos)

    chosen: Optional[Possibility] = None
    best_diff: Optional[int] = None

    def render_cands(cands: Sequence[Possibility], mark_label: str = "best",
                     verified: Optional[Sequence[Possibility]] = None,
                     show_gate: bool = True) -> List[str]:
        """List each candidate's `Options are` line (regime check then length
        diff). `verified` None means every candidate is eligible to win the pick;
        otherwise only candidates in `verified` (by identity) may set the running
        minimum — the rest are shown unmarked (the unknown-middle's unverified
        candidates, pending the brute-force below). `show_gate` False drops the
        per-line `needs<gate>` (the unknown section, uniformly a+c<=7 by its header)."""
        nonlocal chosen, best_diff
        out = ["Options are"]
        for pp in cands:
            eligible = verified is None or any(pp is v for v in verified)
            rights = (pp.right_cands
                      or ([pp.right] if pp.right is not None else [None]))
            for r in rights:
                segs = _score_possibility(pp, r, show_gate)
                if not segs:                 # no shift-derivable cut -> drop candidate
                    continue
                base = _poss_text(*_poss_regs_lens(pp, r))
                line = base
                for seg, diff, ac in segs:
                    mark = ""
                    if eligible and diff is not None and (best_diff is None
                                                          or diff < best_diff):
                        best_diff = diff
                        mark = f" {mark_label}"
                        chosen = replace(pp, right=r,
                                         right_cands=[r] if r is not None else [],
                                         ac=ac)
                    line += seg + mark
                out.append(line)
        if len(out) == 1:                    # section yielded no candidates
            out.append(" none")
        return out

    def render_unknown() -> List[str]:
        """The `three_unknown` `Options are` + `Testing` blocks. Walks the candidate
        sweep in menu order; a candidate worth verifying (`diff < seed`) gets a `test`
        mark. A sum-0 candidate (exact length fit) is gap-tested INLINE right after
        its option line — its bit-column lines printed directly, no `Testing` header —
        and a full match STOPS the section immediately (the rest of the menu is never
        emitted). If no sum-0 matches, the `Now test absolute sum 1 and higher` header
        introduces the remaining testables in ascending-sum order (with `Testing`
        headers), stopping at the first match. `best_diff` (the running 1/2/3-region
        min) gates which candidates are testable and records the winner."""
        nonlocal chosen, best_diff
        # `seed` FREEZES the test threshold carried in from the 1/2/3-region sections
        # so every `test` decision is derivable in reading order (the running
        # `best_diff` is only lowered after a pass, below). `seed` is shown in `stmt`.
        seed = best_diff
        have = "none" if seed is None else str(seed)
        gate = "" if seed is None else f" with absolute sum less than {seed}"
        stmt = (f"Best absolute sum so far is {have} so test passed options{gate} "
                f"on input and output bit columns immediately for absolute sum 0 "
                f"and later for 1 and higher until match")
        out = [stmt, "Options are"]
        if _GATE_UNKNOWN_ON_ZERO_BEST and seed == 0:
            out.append(" none")              # already 0-best -> skip testing
            return out
        rest: List[Tuple] = []               # (diff, order, pp, r, ac) for sum >= 1
        winner = None
        any_option = False
        order = 0
        for pp in candidates:
            if winner is not None:
                break                        # sum-0 match -> truncate the menu
            r = pp.right
            base = _poss_text(*_poss_regs_lens(pp, r))
            for seg, diff, ac in _score_possibility(pp, r, show_gate=False):
                any_option = True
                testable = diff is not None and (seed is None or diff < seed)
                out.append(base + seg + (" test" if testable else ""))
                if testable and diff == 0:   # exact fit: test inline, no header
                    passed, gl = gap_by_id.get(id(pp), (False, []))
                    out += gl
                    if passed:
                        winner = (diff, pp, r, ac)
                        break
                elif testable:
                    rest.append((diff, order, pp, r, ac))
                order += 1
        if not any_option:                   # section yielded no candidates
            out.append(" none")
        # No sum-0 match -> escalate: the remaining testables in ascending-sum order
        # (stable on ties), each its own `Testing` block, stopping at the first match.
        if winner is None and rest:
            out.append("Now test absolute sum 1 and higher")
            for diff, _o, pp, r, ac in sorted(rest, key=lambda t: (t[0], t[1])):
                passed, gl = gap_by_id.get(id(pp), (False, []))
                out.append(f"Testing {pp.left.label} {r.label} {pp.middle_label}")
                out += gl
                if passed:
                    winner = (diff, pp, r, ac)
                    break
        if winner is not None:
            diff, pp, r, ac = winner
            if best_diff is None or diff < best_diff:
                best_diff = diff
                chosen = replace(pp, right=r, right_cands=[r], ac=ac)
        return out

    blocks = [_strip_blanks(preamble)]
    for name, lines, cands in sections:
        if name == "three_unknown":
            blocks.append(_strip_blanks(lines) + render_unknown())
        else:
            stripped = _strip_blanks(lines)
            if not stripped and not cands:   # empty section (e.g. no 3-region
                continue                     # fillers) — skip the bare `none` block
            blocks.append(stripped + render_cands(cands))

    def _winner_line() -> List[str]:
        return [f"Best option is{_applied_text(chosen)}"]

    # Every case opens on the SAME line: the gate, and the current best (its applied
    # regions + absolute sum, or `none`). A winner under the gate ends there; otherwise
    # the line continues `so try …` and the fallback runs.
    cur = (f"{_applied_text(chosen).strip()} with absolute sum {best_diff}"
           if chosen is not None else "none")
    head = (f"Best option needs to have absolute sum less than {_FALLBACK_SUM_GATE} "
            f"and is currently {cur}")
    keep = chosen is not None and best_diff is not None and best_diff < _FALLBACK_SUM_GATE
    if keep:
        if trace is not None:
            trace["fallback_block"] = 0
        blocks.append([head])
    else:
        # UNIFIED fallback. Fires when no section/unknown candidate won, OR the winner
        # overshot the gate (absolute sum >= _FALLBACK_SUM_GATE — a badly-fit corner).
        # One block, in order:
        #   1. same-family shorter candidates: rebuild the recorded pairs on the
        #      longest-best left/right FAMILIES' sub-maximal corner runs, render them
        #      `Options are`-style, and take the FIRST exact (absolute sum 0) hit —
        #      a greedy longest run over-ran a boundary; the shorter corner is exact
        #      (649bebaf, b158ab98, c8730ea5).
        #   2. else the noised XORNOT/ORNOT bestfit: recover the 2-op rule whose middle
        #      a few flips hid, via Shape C GTE (exact) or the Shape A/B contest
        #      (`_bestfit_block(..., combined=True)` — 05b5ffe1).
        #   3. else per-bit fill (`_fallback_fill`; `chosen` stays None).
        L, _ll, R, _rl = _fallback_corners(an)
        lfam = L.family if L is not None else "none"
        rfam = R.family if R is not None else "none"

        def _runs_of(family: str, anchor: int,
                     fb_rule: Optional[ShiftRule]) -> List[ShiftRule]:
            """Every corner run of `family` at `anchor` in display (run) order —
            non-best included — falling back to the single `_fallback_corners` rule
            when the family has no run pool (e.g. Identity isn't a shift family)."""
            runs = [r for r, _s, _l, _f in _all_corner_runs(an, family, anchor)]
            return runs or ([fb_rule] if fb_rule is not None else [])

        # Open the family pair up to ALL pairings (not just the longest-best), listed
        # like the two-region walk: ` <left>│ <right…>`, one line per left run. The
        # true shorter corner is visible here even though it isn't the longest.
        left_runs = _runs_of(lfam, 0, L)
        right_runs = _runs_of(rfam, NB - 1, R)
        rights_disp = " ".join(r.label for r in right_runs) or "none"
        fb = ([f"{head} so try left and right output matches in the longest best "
               f"left right families {lfam} {rfam}"]
              + [f" {lr.label}│ {rights_disp}" for lr in left_runs])
        chosen = None                        # reject the over-gate / absent winner

        # 1) same-family shorter candidates, first exact (sum-0) hit wins.
        opt = ["Options are"]
        pick: Optional[Possibility] = None
        for pp in _shorter_chain_cands(an, sections):
            if (pp.right is None or L is None or R is None
                    or pp.left.family != lfam or pp.right.family != rfam):
                continue
            for r in (pp.right_cands or [pp.right]):
                segs = _score_possibility(pp, r, show_gate=True)
                if not segs:
                    continue
                line = _poss_text(*_poss_regs_lens(pp, r))
                for seg, diff, ac in segs:
                    mark = ""
                    if pick is None and diff == 0:
                        pick = replace(pp, right=r, right_cands=[r], ac=ac)
                        mark = " best"
                    line += seg + mark
                opt.append(line)
                if pick is not None:
                    break
            if pick is not None:
                break
        if len(opt) == 1:
            opt.append(" none")

        if pick is not None:                 # exact shorter corner -> one block
            if trace is not None:
                trace["fallback_block"] = 1
            chosen = pick
            blocks.append(fb + opt + _winner_line())
        else:
            block = fb + opt                 # block 1: gate + shorter-corner search
            bf = _bestfit_block(an, candidates, combined=True, cur=cur)
            if bf is not None:               # block 2: GTE / Shape A-B bestfit
                if trace is not None:
                    trace["fallback_block"] = 2
                chosen, bf_lines = bf
                block += bf_lines + _winner_line()   # one block: no blank before block 2
            else:                            # block 3: per-bit fill
                if trace is not None:
                    trace["fallback_block"] = 3
                block += _strip_blanks(_fallback_lines(an))
            blocks.append(block)

    parts: List[str] = []
    for i, b in enumerate(blocks):
        if i:
            parts += [""]                    # double newline between sections
        parts += b
    return chosen, parts


def _first_corner(entries, longest: int) -> Optional[ShiftRule]:
    """The first tied-best chain reaching the side's longest length."""
    for _name, bests in entries:
        for rule, length in bests:
            if length == longest:
                return rule
    return None


def _fallback_corners(an: Analysis) -> Tuple[Optional[ShiftRule], int,
                                             Optional[ShiftRule], int]:
    """The left/right best corner rules and their (truncated) lengths for the
    no-possibility fallback. Overlapping corners (`ll + rl > NB`) are tiled by
    shortening the shorter side."""
    ll, rl = an.left_longest, an.right_longest
    if ll + rl > NB:
        if rl > ll:
            ll = NB - rl
        else:
            rl = NB - ll
    ll, rl = max(0, min(ll, NB)), max(0, min(rl, NB))
    L = _first_corner(an.section_lefts, an.left_longest) if an.left_longest else None
    R = _first_corner(an.section_rights, an.right_longest) if an.right_longest else None
    return L, ll, R, rl


# A fallback fill: the per-bit plan plus, for each middle bit, the SECTION_ORDER
# candidate survey ``[(section, [rules])]`` and the chosen rule.
_FallbackFill = Tuple[
    List[Optional[object]],                                  # plan
    Optional[ShiftRule], int, Optional[ShiftRule], int,      # L, ll, R, rl
    List[Tuple[int, List[Tuple[str, List[ShiftRule]]], ShiftRule]],  # middle
]


def _fallback_fill(an: Analysis) -> _FallbackFill:
    """No-possibility fallback per-bit fill: lay the left/right best corners onto
    their bits, then fill every remaining bit with the first SECTION_ORDER rule
    that reproduces that output column, defaulting to HIGH (output 1) when none
    does. Only the extrapolated corners carry prediction risk."""
    plan: List[Optional[object]] = [None] * NB
    L, ll, R, rl = _fallback_corners(an)
    if L is not None:
        for b in range(ll):
            plan[b] = L
    if R is not None:
        for b in range(NB - rl, NB):
            if plan[b] is None:
                plan[b] = R
    orec = {name: _expand_symmetric(recs) for name, recs
            in _build_records(an.cols).items()}
    middle: List[Tuple[int, List[Tuple[str, List[ShiftRule]]], ShiftRule]] = []
    for bit in range(NB):
        if plan[bit] is not None:
            continue
        survey: List[Tuple[str, List[ShiftRule]]] = []
        chosen: Optional[ShiftRule] = None
        for name in SECTION_ORDER:
            drop_zero = name in _DROP_ZERO_SHIFT
            cands: List[ShiftRule] = []
            seen: Set[Tuple] = set()
            for r in _candidates_at(orec[name], bit):
                if drop_zero and _is_zero_shift(r):
                    continue
                key = _rule_canon_key(r)
                if key not in seen:
                    seen.add(key)
                    cands.append(r)
            survey.append((name, cands))
            if chosen is None and cands:
                chosen = cands[0]
        if chosen is None:
            chosen = ShiftRule("HIGH", None, None)     # default HIGH
        plan[bit] = chosen
        middle.append((bit, survey, chosen))
    return plan, L, ll, R, rl, middle


def _fallback_lines(an: Analysis) -> List[str]:
    """Narrate the no-possibility fallback per-bit: a header in the ``<left>
    <right> <middle> lengths L R M`` form (middle = ``unknown``, default HIGH),
    then one line per middle bit with each section's reproducing candidates and
    the SECTION_ORDER pick (``so``)."""
    _plan, L, ll, R, rl, middle = _fallback_fill(an)
    left_label = L.label if L is not None else "none"
    right_label = R.label if R is not None else "none"
    lines = [
        "",
        f"Best option is none so try {left_label} {right_label} unknown "
        f"lengths {ll} {rl} {NB - ll - rl} with default HIGH",
    ]
    for bit, survey, chosen in middle:
        checks = [_or_absent(name, [_compact_ops(r, bit) for r in cands])
                  for name, cands in survey]
        lines.append(f"{bit}│" + ", ".join(checks) + f" so {chosen.label}")
    return lines


def _compute_answer(query: str, plan: Sequence[Optional[object]]) -> str:
    return "".join(_apply_to_query(plan[bit], bit, query) for bit in range(NB))


def predict_struct(examples: Sequence[Tuple[str, str]], query: str = ""):
    """`(answer, plan, poss)` — or None if malformed. `answer` is None with no
    query; `plan` is the 8-entry per-bit apply plan (ShiftRule | MultiOp | None);
    `poss` is the chosen first possibility (None when nothing matched)."""
    an = analyze(examples)
    if an is None:
        return None
    poss = _render_and_select(an)[0]
    plan = _assign_regions(poss)[0] if poss is not None else _fallback_fill(an)[0]
    norm_query = _normalize_bits(query) if query else ""
    answer = _compute_answer(norm_query, plan) if norm_query else None
    return answer, plan, poss


def classify_fallback(examples: Sequence[Tuple[str, str]], query: str = "") -> int:
    """Which unified-fallback block `_render_and_select` fires for these examples:
    0=gate winner kept, 1=spurious over-run (shorter corner), 2=GTE/Shape-A-B
    bestfit, 3=per-bit fill, -1=unanalyzable. Runs the SAME `_render_and_select`
    the trainer runs (no `query`/answer needed) so the verdict can't drift from the
    rendered trace; `query` is accepted for signature parity but unused."""
    an = analyze(examples)
    if an is None:
        return -1
    trace: Dict[str, int] = {}
    _render_and_select(an, trace=trace)
    return trace.get("fallback_block", 0)


def predict(examples: Sequence[Tuple[str, str]], query: str = "") -> Optional[str]:
    res = predict_struct(examples, query)
    return res[0] if res else None


# Router entry points — the structured-walk trace plus the boxed answer.

def _format_shift_apply(rule: ShiftRule, bit: int, query: str) -> str:
    """One per-bit ShiftRule Apply derivation: ``<label> = <fn>(IN<col>...) =
    <fn>(<val>...) = <result>``. Constants collapse to ``= 0/1``; Identity has no
    op step; NOT/binaries show their function form."""
    fam = rule.family
    if fam == "LOW":
        return f"{rule.label} = 0"
    if fam == "HIGH":
        return f"{rule.label} = 1"
    p, s = rule.operands_at(bit)
    if fam == "I":
        return f"{rule.label} = IN{p} = {query[p]}"
    if fam == "NOT":
        vp = query[p]
        return f"{rule.label} = NOT IN{p} = NOT {vp} = {_bit_not(vp)}"
    val = _apply_rule_to_query(rule, bit, query)
    a, b = query[p], query[s]
    # Infix rendering keyed on the IN columns: `INp OP INs`. The NOT versions
    # (ANDNOT/ORNOT/XORNOT) negate the second operand with `¬`, so the result is
    # *derived* from the base op rather than read off a memorized truth-table
    # cell — the model was a coin-flip on entries like ORNOT(1,0) without it.
    base = {"AND": "AND", "OR": "OR", "XOR": "XOR",
            "ANDNOT": "AND", "ORNOT": "OR", "XORNOT": "XOR"}[fam]
    neg = "¬ " if fam in ("ANDNOT", "ORNOT", "XORNOT") else ""
    return (f"{rule.label} = IN{p} {base} {neg}IN{s} = "
            f"{a} {base} {neg}{b} = {val}")


def _multiop_label(mop: MultiOp) -> str:
    """The token that heads a MultiOp derivation: shape A→``XNO…``, B→``ONX…``,
    C→``GTE…``, otherwise the clean family name; all suffixed by the present shift
    digits."""
    fam = mop.family
    if mop.shape == "B":
        base = "ONX"
    elif mop.shape == "C":
        base = "GTE"
    elif fam not in ("MAJ", "Ch") and not fam.endswith("-shift"):
        base = "XNO"
    elif fam == "MAJ":
        base = "Majority"
    elif fam == "Ch":
        base = "Choice"
    else:
        # *-shift gate token is uppercase (family stays e.g. "ORNOT-shift").
        base = fam[:-len("-shift")].upper()
    return base + "".join(str(s) for s in (mop.shl, mop.rot, mop.shr)
                          if s is not None)


def _format_multiop_apply(mop: MultiOp, bit: int, query: str) -> str:
    """One per-bit MultiOp Apply derivation: ``<label> = <reduction> = <final>``.
    The reduction is built by `_multiop_rhs` (shared with the Testing renderer);
    ``IN<col>`` columns are bit-substituted under wrap semantics (matches
    ``_multiop_eval_query``). Absent atoms render as ``0``."""
    def atom_col(shift: Optional[int]) -> Optional[int]:
        return None if shift is None else (shift + bit) % NB

    l_col, t_col, r_col = atom_col(mop.shl), atom_col(mop.rot), atom_col(mop.shr)
    in_or_0 = lambda c: f"IN{c}" if c is not None else "0"
    val_or_0 = lambda c: query[c] if c is not None else "0"
    rhs = _multiop_rhs(mop.family, mop.shape,
                       in_or_0(l_col), in_or_0(t_col), in_or_0(r_col),
                       val_or_0(l_col), val_or_0(t_col), val_or_0(r_col))
    return f"{_multiop_label(mop)} = {rhs}"


def _applying_lines(query: str, answer: str,
                    plan: Sequence[Optional[object]]) -> List[str]:
    """The Applying section: the boxed ``Input`` column, per-bit ``Output``
    derivations from the chosen plan, ending on ``The final answer is: <bits>``.
    A bit with no plan entry renders as ``unknown = 0``."""
    lines: List[str] = ["", f"Applying to query {query}", "Input"]
    lines += [f"{i}│{b}" for i, b in enumerate(query)]
    lines.append("Output by first shifting to find IN column then evaluating")
    for bit in range(NB):
        entry = plan[bit]
        if isinstance(entry, MultiOp):
            lines.append(f"{bit}│ {_format_multiop_apply(entry, bit, query)}")
        elif isinstance(entry, ShiftRule):
            lines.append(f"{bit}│ {_format_shift_apply(entry, bit, query)}")
        else:
            assert False, "Unexpected unknown = 0"
            # lines.append(f"{bit}│ unknown = 0")
    lines += ["", f"The final answer is: {answer}"]
    return lines


def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Build a bit-manipulation CoT from the prompt: parse the example pairs and
    query, emit the structured walk (:func:`explain`), then — given a query — the
    Applying section with the boxed answer (:func:`predict_struct`). ``answer`` is
    ground truth; it scores the trace's derived ``predicted``, never builds it."""
    pairs = _PAIR_RE.findall(prompt)
    query_match = _QUERY_RE.search(prompt)
    query = query_match.group(1) if query_match else ""

    trace = explain(pairs, query)
    if trace is None:
        cot_text, predicted = "", None
    else:
        norm_query = _normalize_bits(query) if query else ""
        predicted = plan = None
        if norm_query:
            res = predict_struct(pairs, norm_query)
            if res is not None:
                predicted, plan, _poss = res
        lines = [trace]
        if predicted is not None:
            # `Best option is` already states the applied lengths and the
            # boundary derivation lives in its `Options are` line, so go
            # straight to Applying (no separate truncation narration).
            lines.extend(_applying_lines(norm_query, predicted, plan))
        cot_text = "\n".join(lines)

    return cot_text, {
        "predicted": predicted,
        "correct": predicted is not None and predicted == _normalize_bits(answer),
        "query": query,
        "path": "main",
    }
