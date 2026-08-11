r"""CoT generator for the equation_transform/symbol question type.

Each problem shows several ``AB op CD = RHS`` example lines and a query LHS.
The LHS is always 5 chars (A, B, op, C, D). The hidden rule per operator is
either a structural concatenation (remove the middle symbol -> ABCD, optionally
swap halves -> CDAB) or an arithmetic rule over a hidden symbol->digit mapping.
The arithmetic cryptarithm subtype cannot be solved in the CoT budget, so
we always commit to the concat guess and report it honestly.

Per-example block (one per input example):

    Example i
    AB?CD = RHS              <- verbatim
    AB?CD                    <- lhs alone
    │A││B││?││C││D│ has operator │?│
     RHS                     <- leading space matters (tokenization parity)
    │R││H││S│...
    │A││B││?││C││D│ becomes │R││H││S│...

Then the query block:

    Query
     AB?CD                   <- leading space
    │A││B││?││C││D│ has operator │?│

    Identify examples with operator │?│
    │A││B││?││C││D│ becomes │R││H││S│... (one line per q-op example)

    {conclusion}
    │A││B││?││C││D│
    │P││R││E││D│
     PRED                    <- leading space

The conclusion line is binary:
* default (forward concat / unconfirmed): ``We should give our best guess.``
* reverse confirmed (swap and remove): ``We should swap and remove the middle symbol.``

The wrapper in ``nemotron/prompts.py`` appends ``</think>`` and the final
answer line.

Contract (see ``nemotron/data/main_router.py``):

    generate_cot(prompt: str, answer: str, **kwargs)
        -> (cot_text: str, meta: Dict[str, Any])

``meta`` always contains:

* ``predicted``: str -- best-effort concat guess (ABCD or CDAB). The
  generator asserts the query LHS is 5 chars, so this is always populated.
* ``correct``:  bool -- ``predicted == answer.strip()``.
* ``is_guess``: bool -- True when q-op examples did NOT structurally
  confirm the chosen rule. Diagnostic only; does NOT gate ``predicted``.

Plus diagnostic keys: ``query_op``, ``query_op_examples``, ``chosen_rule``.
"""

import re
from typing import Any, Dict, List, Tuple


# LHS layout: A=0, B=1, op=2, C=3, D=4.
_CONCAT_FWD: Tuple[int, ...] = (0, 1, 3, 4)
_CONCAT_REV: Tuple[int, ...] = (3, 4, 0, 1)

# Emit the tokenizer-merge representation steps (the ``_tok_segments``
# multi-symbol-unit lines, e.g. ``│#{││*"││!│``). These assert a BPE grouping
# that is context/space-dependent and inconsistent with the bare answer's
# tokenization (the EX/Q_COMPUTE corruption source). True = current behaviour;
# False = every chain step is single-symbol-per-│…│ (tokenizer-consistent,
# ~30% shorter). Single A/B toggle — flip and regenerate.
_EMIT_TOKEN_MERGE_STEPS: bool = True
_EMIT_SUFFIX: bool = True


def _apply_structural(lhs: str, positions: Tuple[int, ...]) -> str:
    return "".join(lhs[p] for p in positions)


def _seg(*parts: str) -> str:
    """Wrap each segment with │…│ and concatenate."""
    return "".join(f"│{p}│" for p in parts)


def _tok_segments(tokenizer: Any, s: str) -> List[str]:
    """Split *s* into the tokenizer's token pieces (decoded), or per-char.

    Encodes *s* then decodes each id individually so the pieces are
    human-readable and ``"".join(pieces) == s`` (leading space preserved).
    Falls back to one segment per character when no tokenizer is supplied
    or the per-token decode does not round-trip — keeps the chain
    well-formed for the script call sites that pass no tokenizer.
    """
    if tokenizer is None:
        return list(s)
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
        segs = [tokenizer.decode([int(i)]) for i in ids]
    except Exception:
        return list(s)
    if not segs or "".join(segs) != s:
        return list(s)
    return segs


def _suffix_combines(tokenizer: Any, s: str, suffix: str) -> bool:
    """True iff *suffix* fuses into *s*'s trailing token under the
    tokenizer rather than standing as its own piece.

    Threading the suffix only buys last-token parity when the value's
    real right-neighbor actually merges with its terminal symbol; if it
    tokenizes standalone, threading it just injects a literal newline
    mid-chain for zero parity gain (the reason ``_EMIT_SUFFIX`` defaults
    off). Test: tokenize *s* vs ``s + suffix`` -- no merge iff the
    extended split is exactly the base split plus a lone *suffix* piece.
    The per-char fallback in ``_tok_segments`` (no tokenizer /
    non-round-trip) makes this ``False`` -- unknown ⇒ don't risk the
    line break.
    """
    if not suffix or tokenizer is None:
        return False
    return (
        _tok_segments(tokenizer, s + suffix)
        != _tok_segments(tokenizer, s) + [suffix]
    )


def _concat_chain(
    lhs: str,
    tokenizer: Any = None,
    prefix: str = "",
    suffix: str = "",
    tail: bool = True,
    reverse: bool = False,
) -> str:
    """Concat derivation, tokenizer-aware. Forward by default; reverse with
    ``reverse=True``.

    Forward path:
        full expr → tokenized → per-char → drop op + leading space (chars
        individual, space rides the first char) → recombine by tokenization
        → full expr w/ delimiter → bare expression.

    Reverse path (the LHS-echo + op-removed steps carry a ``" reversed"``
    tag; the tag drops at the swap, then the chain continues like the
    forward path on the swapped string):
        full expr [rev] → tokenized [rev] → per-char [rev] → drop op
        [rev] → swap halves → add leading space → recombine by
        tokenization → full expr w/ delimiter → bare expression.

    Token-parity threading. *prefix* / *suffix* are the value's true
    left / right neighbors in its anchor line (for the query: it sits in
    ``determine the result for: <lhs>\\n`` → ``prefix=" "``, ``suffix="\\n"``;
    for examples the LHS is at line start → defaults ""):

    * *prefix* rides the first char of **every** LHS-echo step (verbatim,
      tokenized, per-char) so the first token id matches the anchor — not
      just the first step. The result-side steps keep their own fixed " "
      (the answer is always emitted as ``... is: <ans>`` — that leading
      space is intrinsic to the answer slot, not the query context).
    * *suffix* attaches to the **terminal** bare term only — the single
      place a real right-neighbor can sit without a ``\\n`` splitting the
      one-line chain. Pass "" when the chain is followed by a
      newline-joined ``parts`` entry: that newline already supplies the
      terminal's right context, so last-token parity holds structurally
      and appending here would double it.
    """
    a, b, op, c, d = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
    fwd = a + b + c + d              # op removed, pre-swap
    result = c + d + a + b if reverse else fwd
    tag = " reversed" if reverse else ""

    terms: List[str] = [_seg(prefix + lhs + suffix) + tag + " "]  # "| ABCDE| reversed ", in examples no space before A
    if _EMIT_TOKEN_MERGE_STEPS:
        terms.append(
            " " + _seg(*_tok_segments(tokenizer, prefix + lhs + suffix)) + tag + " "
        )  # " | AB||CDE| reversed "
    terms.append(
        " " + _seg(prefix + lhs[0], *lhs[1:-1], lhs[-1] + suffix) + tag + " "
    )  # " | A||B||C||D||E| reversed "
    if prefix != "":
        terms.append(" " + _seg(*lhs) + tag + " ")  # " |A||B||C||D||E| reversed "
    terms.append(" " + _seg(*fwd) + tag + " ")  # " |A||B||D||E| reversed "
    if reverse:
        terms.append(" " + _seg(*result) + " ")  # " |D||E||A||B| "
    terms.append(
        " " + _seg(" " + result[0], *result[1:-1], result[-1] + suffix) + " "
    )  # " | D||E||A||B| "
    if _EMIT_TOKEN_MERGE_STEPS:
        terms.append(
            " " + _seg(*_tok_segments(tokenizer, " " + result + suffix)) + " "
        )  # " | DE||AB| "
    terms.append(" " + _seg(" " + result + suffix) + " ")  # " | DEAB| "
    terms.append(" " + result + " ")  # " DEAB "
    if tail:
        if _EMIT_TOKEN_MERGE_STEPS:
            terms.append(
                " " + _seg(*_tok_segments(tokenizer, " " + result)) + " "
            )  # " | DE||AB| "
        terms.append(
            " " + _seg(" " + result[0], *result[1:]) + " "
        ) # " | D||E||A||B| "
        terms.append(" " + _seg(*result))  # " |D||E||A||B|"
    else:
        terms[-1] = terms[-1][:-1]  # Get rid of last space

    return "=".join(terms)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_QUERY_RE = re.compile(r"determine the result for:\s*(.+)", re.IGNORECASE)


def _parse_prompt(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """Return ``(examples, query_lhs)``.

    Each example is ``(lhs, rhs)`` with ``lhs`` a 5-char string. ``query_lhs``
    is the 5-char query. Missing pieces come back as empty list / empty str.
    """
    examples: List[Tuple[str, str]] = []
    query_lhs: str = ""
    for ln in prompt.splitlines():
        qm = _QUERY_RE.search(ln)
        if qm:
            q = qm.group(1).strip().replace(" ", "")
            if len(q) >= 5:
                query_lhs = q[:5]
            continue
        if "=" in ln:
            low = ln.lstrip().lower()
            if low.startswith(("in alice", "please put")):
                continue
            lhs, _, rhs = ln.partition("=")
            lhs = lhs.strip().replace(" ", "")
            rhs = rhs.strip()
            if len(lhs) == 5:
                examples.append((lhs, rhs))
    return examples, query_lhs


# ---------------------------------------------------------------------------
# Per-example classification + decision
# ---------------------------------------------------------------------------

_CLS_FWD = "concatenation"
_CLS_REV = "concatenation reversed"
_CLS_UNKNOWN = "unknown"
_DECISION_NONE = "none"


def _classify(lhs: str, rhs: str) -> str:
    """Classify a single q-op example as concat / rev concat / unknown."""
    if _apply_structural(lhs, _CONCAT_FWD) == rhs:
        return _CLS_FWD
    if _apply_structural(lhs, _CONCAT_REV) == rhs:
        return _CLS_REV
    return _CLS_UNKNOWN


def _decide(q_op_exs: List[Tuple[str, str]]) -> str:
    """Return ``concatenation``, ``rev concatenation``, ``unknown``, or ``none``.

    * ``none`` -- no q-op examples (default to concatenation when emitting).
    * ``concatenation`` -- every q-op example matches ABCD.
    * ``rev concatenation`` -- every q-op example matches CDAB.
    * ``unknown`` -- mixed or any unclassifiable q-op example.
    """
    if not q_op_exs:
        return _DECISION_NONE
    cls = [_classify(l, r) for l, r in q_op_exs]
    if all(c == _CLS_FWD for c in cls):
        return _CLS_FWD
    if all(c == _CLS_REV for c in cls):
        return _CLS_REV
    return _CLS_UNKNOWN


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cot(
    prompt: str,
    answer: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    tokenizer = kwargs.get("tokenizer")
    examples, query_lhs = _parse_prompt(prompt)
    answer_stripped = (answer or "").strip()
    assert len(query_lhs) == 5, (
        f"query LHS must be 5 chars (A B op C D), got {query_lhs!r} "
        f"from prompt: {prompt!r}"
    )

    q_op = query_lhs[2]
    q_op_exs = [(l, r) for l, r in examples if l[2] == q_op]
    decision = _decide(q_op_exs)
    if decision == _CLS_FWD:
        predicted_raw = _apply_structural(query_lhs, _CONCAT_FWD)
        is_guess = False
    elif decision == _CLS_REV:
        predicted_raw = _apply_structural(query_lhs, _CONCAT_REV)
        is_guess = False
    elif decision == _CLS_UNKNOWN:
        # First symbol of the first example overall (== alphabet[0]).
        predicted_raw = examples[0][0][0]
        is_guess = True
    else:  # _DECISION_NONE
        predicted_raw = _apply_structural(query_lhs, _CONCAT_FWD)
        is_guess = True

    parts: List[str] = [
        "We need to infer transformation rules from examples. "
        "The examples involve strings of symbols. "
        "We need to know the transformations. "
        "They are concatenation, concatenation reversed, and unknown. "
        "Let's list examples."
    ]

    for i, (lhs, rhs) in enumerate(examples, start=1):
        op = lhs[2]
        cls = _classify(lhs, rhs)
        parts.append("")
        parts.append(f"Example {i}")
        parts.append(f"{lhs} = {rhs}")  # "`!*[{ = '?[`"

        # Suffix threads into the rhs/target terminal here (" " + rhs +
        # suffix). Emit it only when it actually fuses with rhs's last
        # token -- otherwise it just breaks the one-line chain.
        suffix = (
            '\n'
            if _EMIT_SUFFIX and _suffix_combines(tokenizer, " " + rhs, '\n')
            else ''
        )

        # Target derivation
        target_tail = [" " + _seg(" " + rhs + suffix) + " "]
        if _EMIT_TOKEN_MERGE_STEPS:
            target_tail.append(
                " " + _seg(*_tok_segments(tokenizer, " " + rhs + suffix)) + " "
            )
        target_tail.append(
            " " + (_seg(" " + rhs[0], *rhs[1:-1], rhs[-1] + suffix) if rhs else _seg()) + " "
        )
        target_tail.append(
            " " + _seg(*rhs)
        )
        parts.append(
            "The target is " + rhs + "\n=" + "=".join(target_tail))

        fwd_res = _apply_structural(lhs, _CONCAT_FWD)
        rev_res = _apply_structural(lhs, _CONCAT_REV)
        fwd_match = "Yes." if fwd_res == rhs else "No."
        rev_match = "Yes." if rev_res == rhs else "No."
        # Echo both sides in the same separated form the chains terminate in,
        # so the verdict is a visible │…│-vs-│…│ comparison.
        tgt_sep = _seg(*rhs)
        parts.append(
            f"- concatenation? {_concat_chain(lhs, tokenizer)}")
        parts.append(
            f"- target {tgt_sep} and concatenation {_seg(*fwd_res)} "
            f"match? {fwd_match}")
        parts.append(
            f"- concatenation reversed? "
            f"{_concat_chain(lhs, tokenizer, reverse=True)}")
        parts.append(
            f"- target {tgt_sep} and concatenation reversed "
            f"{_seg(*rev_res)} match? {rev_match}")
        parts.append(
            f"So what is the operator? {_seg(*lhs)} has operator │{op}│. "
            f"Is │{op}│ concatenation, concatenation reversed, or unknown? "
            f"Probably {cls}."
        )

    parts.append("")
    parts.append(f"Now we need to determine the result for: {query_lhs}")
    # query_lhs sits in "...for: <lhs>\n": left neighbor is the space after
    # "for:", right neighbor is the newline before "- concatenation?".
    # prefix=" " threads that space through every LHS-echo step. The "\n"
    # right-neighbor is worth threading as suffix ONLY when it fuses with
    # the (prefix-anchored) query_lhs terminal token under the tokenizer;
    # if it tokenizes standalone, the chain's terminal already gets its
    # "\n" from the parts-join, so parity holds structurally and threading
    # would only inject a literal newline mid-chain.
    suffix = (
        '\n'
        if _EMIT_SUFFIX and _suffix_combines(tokenizer, ' ' + query_lhs, '\n')
        else ''
    )
    fwd = _concat_chain(query_lhs, tokenizer, prefix=' ', suffix=suffix, tail=False)
    parts.append(f"- concatenation? {fwd}")
    back = _concat_chain(query_lhs, tokenizer, prefix=' ', suffix=suffix, tail=False, reverse=True)
    parts.append(f"- concatenation reversed? {back}")
    if examples:
        ex_lhs = examples[0][0]
        first_sym = ex_lhs[0]
        parts.append(
            f"- unknown? The first symbol of first example "
            f"{_seg(ex_lhs)} is │{first_sym}│ = {first_sym}"
        )
    if q_op_exs:
        parts.append(
            f"Is │{q_op}│ present in the examples? Yes. "
            f"So is │{q_op}│ concatenation, concatenation reversed, or "
            f"unknown? Probably {decision}. "
            f"So the answer is: {predicted_raw}"
        )
    else:
        parts.append(
            f"Is │{q_op}│ present in the examples? Not. "
            f"So we can guess concatenation. "
            f"So the answer is: {predicted_raw}"
        )
    cot = "\n".join(parts)

    return cot, {
        "predicted": predicted_raw,
        "correct": predicted_raw == answer_stripped,
        "query_op": q_op,
        "query_op_examples": len(q_op_exs),
        "chosen_rule": decision,
        "is_guess": is_guess,
    }