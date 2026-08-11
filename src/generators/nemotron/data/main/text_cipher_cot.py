"""CoT generator for the text_cipher question type.

Each problem gives a few ``cipher_sentence -> plain_sentence`` examples under a
hidden bijective lowercase-letter substitution cipher, then asks the model to
decrypt a new query phrase.

New contract (see ``nemotron/data/main_router.py``):

    generate_cot(prompt: str, answer: str, **kwargs)
        -> (cot_text: str, meta: Dict[str, Any])

``meta`` always contains:

* ``predicted``: Optional[str] — the generator's genuine derivation from the
  prompt. Populated with the decrypted query string where unknown letters are
  best-effort filled in (see below). Never populated from ``answer``.
* ``correct``:  Optional[bool] — ``predicted == answer.strip()``.

Output structure (mirrors ``DATA_REFACTOR_PLAN.md``'s "Text cipher" section):

    We need to figure out substitution cipher mapping from examples. Let's
    compile mapping from cipher letters to plain letters.

    Example 1
    <cipher_sentence> -> <plain_sentence>
    <cipher_word_0> -> <plain_word_0>
    <spaced cipher word> -> <spaced plain word>
    <c> -> <p>                          # one per letter position (duplicates kept)
    ...
     <cipher_word_k> -> <plain_word_k>  # leading space for word index >= 1
    ...

    Mapping
    a -> ?
    b -> ?
    ...                                 # always a..z; '?' for unknown so far

    Example 2
    <repeat>

    Mapping
    <updated with new inferences>

    Thus we have the cipher. Decrypt for <query_cipher>.
    <spaced cipher word> -> <spaced plain word> -> <plain_word>
     <spaced cipher word> -> <spaced plain word> -> <plain_word>  # leading space
    ...

    Thus the decrypted phrase is "<with ? for unknowns>". Probably the answer
    is "<best-effort resolution>". That seems plausible.

    Thus the output is <final answer>.

Deviations from the literal plan:

* The plan's sample shows an inconsistent leading-space convention for the
  per-word ``cipher_word -> plain_word`` summary lines inside an example block
  (``apgzahmd -> imagines`` has no leading space but `` yiit -> book`` does).
  We use a consistent rule: word index 0 has no leading space, all later words
  get exactly one leading space. This matches the plan's own rule in the
  decryption block.
"""

import hashlib
import random
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from nemotron.data.main.text_cipher_syn import (
    _ADJECTIVE,
    _LOCATION,
    _OBJECT,
    _PREP,
    _SUBJECT,
    _VERB,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_EXAMPLE_RE = re.compile(r"^(\S.*?)\s+->\s+(\S.*?)\s*$")
_QUERY_RE = re.compile(
    r"Now,\s*decrypt the following text:\s*(.+?)\s*$",
    re.DOTALL,
)

def _parse_examples(prompt: str) -> List[Tuple[str, str]]:
    """Return ``[(cipher_sentence, plain_sentence), ...]`` in prompt order.

    Examples are any lines of the form ``<X> -> <Y>`` between the header and
    the ``Now, decrypt ...`` line. Lines that don't match are skipped.
    """
    examples: List[Tuple[str, str]] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("now,"):
            break
        m = _EXAMPLE_RE.match(stripped)
        if m:
            examples.append((m.group(1), m.group(2)))
    return examples


def _parse_query(prompt: str) -> str:
    m = _QUERY_RE.search(prompt)
    if not m:
        raise ValueError("Could not find query phrase in text_cipher prompt.")
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Mapping accumulation
# ---------------------------------------------------------------------------

def _extend_mapping(
    mapping: Dict[str, str],
    cipher_sentence: str,
    plain_sentence: str,
) -> Dict[str, str]:
    """Return a NEW mapping extended with inferences from one example.

    Aligns ``cipher_sentence`` and ``plain_sentence`` character-by-character.
    Only alphabetic cipher characters are recorded. Conflicts (a cipher letter
    already mapped to a different plain letter) are left at the existing value
    — the first assignment wins, mirroring a student filling in a table.
    """
    out = dict(mapping)
    for c, p in zip(cipher_sentence, plain_sentence):
        if not c.isalpha():
            continue
        if c in out:
            continue
        out[c] = p
    return out


def _final_mapping(examples: List[Tuple[str, str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for cipher_sentence, plain_sentence in examples:
        mapping = _extend_mapping(mapping, cipher_sentence, plain_sentence)
    return mapping


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _spaced(s: str) -> str:
    """Insert a single space between each character of ``s``."""
    return " ".join(list(s))


def _numbered(s: str) -> str:
    """Spelled-out + 1-indexed positions: ``hatter`` -> ``h 1 a 2 t 3 t 4 e 5 r 6``."""
    return "0 " + " ".join(f"{ch} {i + 1}" for i, ch in enumerate(s))


def _prefix(idx: int) -> str:
    """Leading space convention: word index 0 no space, otherwise one space."""
    return "" if idx == 0 else " "


def _render_example_block(
    cipher_sentence: str,
    plain_sentence: str,
    cumulative_mapping_before: Dict[str, str],
) -> Tuple[str, Dict[str, str]]:
    """Render one example block and return the updated cumulative mapping."""
    cipher_words = cipher_sentence.split(" ")
    plain_words = plain_sentence.split(" ")

    lines: List[str] = [f"{cipher_sentence} -> {plain_sentence}"]
    for i, (cw, pw) in enumerate(zip(cipher_words, plain_words)):
        pref = _prefix(i)
        lines.append(f"{pref}{cw} -> {pw}")
        lines.append(f"{_spaced(cw)} -> {_spaced(pw)}")
        lines.append("So " + ", ".join(f"{c} -> {p}" for c, p in zip(cw, pw)))

    new_mapping = _extend_mapping(
        cumulative_mapping_before, cipher_sentence, plain_sentence
    )
    return "\n".join(lines), new_mapping


def _render_mapping_block(cumulative_mapping: Dict[str, str]) -> str:
    """Render the ``Mapping`` block listing a..z with '?' for unknowns."""
    pairs = [
        f"{chr(ord('a') + i)} -> {cumulative_mapping.get(chr(ord('a') + i), '?')}"
        for i in range(26)
    ]
    return "Thus mapping: " + ", ".join(pairs)


def _render_inverse_mapping_block(cumulative_mapping: Dict[str, str]) -> str:
    """Render the ``Inverse mapping`` block: plain letter -> cipher letter(s).

    Iterates plain letters a..z; lists all cipher letters mapping to each
    (alphabetical, comma-separated) or ``?`` if none. A trailing
    ``So unknown letters are ...`` line lists cipher letters whose plain side
    is still unknown."""
    inverse: Dict[str, List[str]] = {}
    for i in range(26):
        c = chr(ord("a") + i)
        p = cumulative_mapping.get(c, "?")
        inverse.setdefault(p, []).append(c)
    lines: List[str] = ["Inverse mapping"]
    for i in range(26):
        p = chr(ord("a") + i)
        ciphers = sorted(inverse.get(p, []))
        if ciphers:
            lines.append(f"{p} -> {', '.join(ciphers)}")
        else:
            lines.append(f"{p} -> ?")
    unknowns = sorted(inverse.get("?", []))
    if unknowns:
        lines.append(f"So missing letters are: {', '.join(unknowns)}")
    return "\n".join(lines)


def _render_unknowns_line(
    used_plain: "set[str]",
    output_letters: List[str],
    first_arrow: Optional[Tuple[str, str]] = None,
) -> str:
    """Return ``Now the output letters are: ...`` prose listing the supplied
    ``output_letters`` (insertion order, no ``?``), then a comma-separated
    per-letter ``a is mapped, b is not, ...`` summary, then either
    ``So missing letters are <unused plain letters>`` or
    ``So missing letters are none``.

    ``first_arrow`` is the (cipher, plain) pair for the first mapped letter
    in cipher order; surfaces in the parenthetical that anchors the listing
    to the mapping line and gives the model a high-prob bridge into the
    first plain letter."""
    parts: List[str] = []
    unused: List[str] = []
    for i in range(26):
        p = chr(ord("a") + i)
        if p in used_plain:
            parts.append(f"{p} is mapped")
        else:
            parts.append(f"{p} is not")
            unused.append(p)
    summary = (
        f"So missing letters are: {', '.join(unused)}."
        if unused
        else "So missing letters are: none."
    )
    anchor = (
        f" (the right side of arrow starting from {first_arrow[0]} -> {first_arrow[1]})"
        if first_arrow is not None
        else ""
    )
    return (
        f"Now the output letters in final mapping are{anchor}: "
        f"{', '.join(output_letters)}. "
        f"So which letters are missing? {', '.join(parts)}. "
        f"{summary}"
    )


def _resolve_unknowns(mapping: Dict[str, str]) -> Dict[str, str]:
    """Best-effort fill-in using process of elimination.

    If exactly one cipher letter is unmapped AND exactly one plain letter is
    unused in the range of the current mapping, the unmapped cipher letter
    must decode to the unused plain letter — fill it in. Otherwise return a
    copy unchanged.
    """
    resolved = dict(mapping)
    unmapped_cipher = [
        chr(ord("a") + i)
        for i in range(26)
        if chr(ord("a") + i) not in resolved
    ]
    used_plain = {
        v for v in resolved.values()
        if isinstance(v, str) and len(v) == 1 and v.isalpha()
    }
    unused_plain = [
        chr(ord("a") + i)
        for i in range(26)
        if chr(ord("a") + i) not in used_plain
    ]
    if len(unmapped_cipher) == 1 and len(unused_plain) == 1:
        resolved[unmapped_cipher[0]] = unused_plain[0]
    return resolved


# Closed-vocabulary prior over query words, derived from text_cipher_syn's
# sentence templates and slot lists. Each syn template is sampled with P=0.25
# and fills its slots uniformly, so the expected count per query sentence of a
# given word is sum over slots where it appears of P(slot included) / |slot|.
# Weights below are expected counts × 10000 (rounded to int for readability).
# Matches the empirical distribution in csvs/train.csv within sampling noise.
def _build_vocab() -> Dict[str, int]:
    # P(slot filled) per sentence given the 4-template uniform mix:
    #   T1 S V O;  T2 S V PREP L;  T3 the ADJ S V;  T4 S V the ADJ O
    slot_prob: List[Tuple[List[str], float]] = [
        (list(_SUBJECT),   1.00),  # every template
        (list(_VERB),      1.00),  # every template
        (list(_OBJECT),    0.50),  # T1 + T4
        (list(_LOCATION),  0.25),  # T2 only
        (list(_PREP),      0.25),  # T2 only
        (list(_ADJECTIVE), 0.50),  # T3 + T4
    ]
    weights: Dict[str, float] = {"the": 0.50}  # T3 + T4
    for words, p in slot_prob:
        per_word = p / len(words)
        for w in words:
            weights[w] = weights.get(w, 0.0) + per_word
    return {w: int(round(f * 10000)) for w, f in weights.items()}


_VOCAB: Dict[str, int] = _build_vocab()

_VERB_SET: FrozenSet[str] = frozenset(_VERB)
_PREP_SET: FrozenSet[str] = frozenset(_PREP)
_ADJECTIVE_SET: FrozenSet[str] = frozenset(_ADJECTIVE)

_TEMPLATE_FORM: Dict[int, str] = {
    1: '"subject" "verb" "object"',
    2: '"subject" "verb" "preposition" "location"',
    3: '"the" "adjective" "subject" "verb"',
    4: '"subject" "verb" "the" "adjective" "object"',
}

_NUM_WORDS: Dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

_ORDINAL_WORDS: Dict[int, str] = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}


def _word_pos(word: str) -> str:
    """Return the part-of-speech label for a plain vocabulary word."""
    if word == "the":
        return "a definite article"
    if word in _VERB_SET:
        return "a verb"
    if word in _PREP_SET:
        return "a preposition"
    if word in _ADJECTIVE_SET:
        return "an adjective"
    return "a noun"


def _detect_template(plain_words: List[str]) -> int:
    """Return 1-4 matching the sentence template used."""
    if not plain_words:
        return 1
    if plain_words[0] == "the":
        return 3
    if len(plain_words) >= 3 and plain_words[2] in _PREP_SET:
        return 2
    if len(plain_words) >= 3 and plain_words[2] == "the":
        return 4
    return 1


def _render_word_analysis(displayed_words: List[str]) -> List[str]:
    """Return CoT lines for the word-by-word summary section."""
    lines: List[str] = [
        "So result is",
    ]
    for word in displayed_words:
        lines.append(_spaced(word))
    return lines


def _render_output_letters_block(
    used_plain: "set[str]",
    output_letters: List[str],
    first_arrow: Optional[Tuple[str, str]] = None,
) -> List[str]:
    """Return the output letter inventory.

    ``output_letters`` is the ordered list of plain letters that will appear
    in the decoded output (insertion order, deduped, ``?`` excluded)."""
    return [_render_unknowns_line(used_plain, output_letters, first_arrow)]


_NEIGHBOR_MAX_DIST: int = 5


def _levenshtein_capped(a: str, b: str, cap: int) -> int:
    """Levenshtein distance between ``a`` and ``b``, capped at ``cap + 1``
    (any return value of ``cap + 1`` means ``> cap``)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > cap:
            return cap + 1
        prev = cur
    return min(prev[lb], cap + 1)


_ROLE_TO_CATEGORY: Dict[str, FrozenSet[str]] = {
    "subject": frozenset(_SUBJECT),
    "verb": frozenset(_VERB),
    "object": frozenset(_OBJECT),
    "location": frozenset(_LOCATION),
    "preposition": frozenset(_PREP),
    "adjective": frozenset(_ADJECTIVE),
}


def _render_role_matchings(
    role: str,
    dword: str,
    used_plain: "set[str]",
    position: int = 1,
    phrase_so_far: str = "",
) -> str:
    """Render positional-match counts for ``dword`` against every word in
    ``role``'s category (alphabetical order), marking each new running maximum.

    ``dword`` may contain ``?`` for letters not yet resolved; those positions
    are skipped when counting matches.  Falls back to a one-liner for the
    ``the`` pseudo-role which has no category. ``position`` is the 1-indexed
    word position used in the ordinal-prefixed header line."""
    cat = _ROLE_TO_CATEGORY.get(role)
    if cat is None:
        ordinal_pre = _ORDINAL_WORDS.get(position, f"{position}th")
        return (
            f'The {ordinal_pre} word has role "{role}". '
            f"Thus phrase so far is: {phrase_so_far}"
        )

    candidates = sorted(cat)
    ordinal = _ORDINAL_WORDS.get(position, f"{position}th")
    article = "an" if role[:1] in "aeiou" else "a"
    n = len(dword)
    # Acceptable lengths: target-1, target, target+1 (skip non-positive).
    lengths = [L for L in (n - 1, n, n + 1) if L > 0]
    length_list = ", ".join(str(L) for L in lengths)
    match_phrase = ", ".join(f"{L} matches" for L in lengths)
    length_clause = (
        f"But wait, we might be off by one: let's match {length_list} else no match. "
        f"So {match_phrase}."
    )
    # Distinct known letters in dword (insertion order, ``?`` excluded).
    known_letters: List[str] = []
    seen: "set[str]" = set()
    for ch in dword:
        if ch != "?" and ch not in seen:
            seen.add(ch)
            known_letters.append(ch)
    known_set = set(known_letters)
    unused = [
        chr(ord("a") + i)
        for i in range(26)
        if chr(ord("a") + i) not in used_plain
    ]
    missing_set = set(unused)
    known_str = ", ".join(known_letters) if known_letters else "none"
    missing_str = ", ".join(unused) if unused else "none"
    lines: List[str] = [
        f'The {ordinal} word has role "{role}" '
        f'and has "{_numbered(dword)}" {n} letters. '
        f"{length_clause} "
        f"Let's count points. "
        f"Start with 1 point if the length matches {n} else 0 points. "
        f"Assign 10 points for matching letters: {known_str}. "
        f"Assign 1 point for missing letters: {missing_str}. "
        f'Let\'s track the max for the role "{role}".'
    ]
    running_max = -1
    winning_word: Optional[str] = None
    cand_counts: List[Tuple[str, int]] = []

    for cand in candidates:
        head = f"- {cand} {len(cand)}"
        if abs(len(dword) - len(cand)) <= 1:
            running = 1 if len(cand) == len(dword) else 0
            trace_parts: List[str] = [str(running)]
            for ch in cand:
                if ch in known_set:
                    running += 10
                elif ch in missing_set:
                    running += 1
                trace_parts.append(ch)
                trace_parts.append(str(running))
            trace = " ".join(trace_parts)
            if running > running_max:
                running_max = running
                winning_word = cand
            cand_counts.append((cand, running))
            head += f" matches. {trace} => max is {running_max} for {winning_word}."
        lines.append(head)

    lines.append(
        f"So max is {running_max} for {winning_word}. "
        f"So the true next word by scoring is that exactly: {winning_word}. "
        f"Thus the next word in phrase is {winning_word}. "
        f"Thus phrase so far is: {phrase_so_far}"
    )
    return "\n".join(lines)


def _pick_foil(gold: str, role: str) -> Optional[str]:
    """Same-category word with the smallest Levenshtein distance to ``gold``
    (ties broken alphabetically). Returns ``None`` when the role has no other
    same-category words (e.g. ``the``)."""
    cat = _ROLE_TO_CATEGORY.get(role)
    if cat is None:
        return None
    candidates = sorted(w for w in cat if w != gold)
    if not candidates:
        return None
    return min(candidates, key=lambda w: _levenshtein_capped(gold, w, 100))


def _choose_foil_guess(
    predicted: str,
    confusion_p: float,
    rng: random.Random,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Maybe replace one word of ``predicted`` with a uniform same-category
    foil, modelling a plausible mis-guess (e.g. "valley" vs "map") that the
    scoring section then corrects back to the truth.

    With probability ``confusion_p`` exactly one eligible word is swapped: pick
    one role uniformly from the phrase's template (excluding ``the`` and any
    role whose decoded word still contains unknowns), then pick a uniform
    same-category word other than the true one. Returns
    ``(guess_phrase, foil_word, role)``; when no confusion is applied
    ``guess_phrase == predicted`` and the latter two are ``None``.

    Only the displayed initial guess changes — the true ``predicted`` and the
    per-role scoring downstream are untouched, so the final answer stays correct.
    """
    if confusion_p <= 0.0:
        return predicted, None, None
    words = predicted.split(" ")
    template = _detect_template(words)
    eligible = [
        (role, idx)
        for role, idx in _roles_for_template(template)
        if role in _ROLE_TO_CATEGORY and idx < len(words) and "?" not in words[idx]
    ]
    if not eligible or rng.random() >= confusion_p:
        return predicted, None, None
    role, idx = rng.choice(eligible)
    gold = words[idx]
    alternatives = sorted(w for w in _ROLE_TO_CATEGORY[role] if w != gold)
    if not alternatives:
        return predicted, None, None
    foil = rng.choice(alternatives)
    words[idx] = foil
    return " ".join(words), foil, role


def _roles_for_template(template: int) -> List[Tuple[str, int]]:
    if template == 1:
        return [("subject", 0), ("verb", 1), ("object", 2)]
    if template == 2:
        return [("subject", 0), ("verb", 1), ("preposition", 2), ("location", 3)]
    if template == 3:
        return [("the", 0), ("adjective", 1), ("subject", 2), ("verb", 3)]
    return [
        ("subject", 0), ("verb", 1), ("the", 2), ("adjective", 3), ("object", 4),
    ]


def _render_vocab_block() -> List[str]:
    """Return the wonderland vocabulary listing lines (shared with practice)."""
    def _with_lengths(words: List[str]) -> str:
        return ", ".join(f"{w} {len(w)}" for w in words)
    return [
        "We need to know the dictionary words used in phrase.\n"
        f'- subject: {_with_lengths(_SUBJECT)}.',
        f'- object: {_with_lengths(_OBJECT)}.',
        f'- location: {_with_lengths(_LOCATION)}.',
        f'- verb: {_with_lengths(_VERB)}.',
        f'- adjective: {_with_lengths(_ADJECTIVE)}.',
        f'- preposition: {_with_lengths(_PREP)}.',
    ]


def _render_structures_block() -> List[str]:
    """Return the sentence-structure listing — one bullet per template in
    ``_TEMPLATE_FORM`` (kept in sync with ``text_cipher_syn`` templates),
    each annotated with its word count."""
    lines: List[str] = ["We need to know the grammar used for phrase."]
    for i in sorted(_TEMPLATE_FORM):
        form = _TEMPLATE_FORM[i]
        n = len(form.split(" "))
        lines.append(f"- {form} -> {n} words.")
    return lines


def _render_match_block(
    displayed_words: List[str],
    predicted: str,
    used_plain: "set[str]",
    guess: Optional[str] = None,
) -> List[str]:
    """Return the per-word display + ``We have N words. So grammar
    likely is ...`` summary that bridges the structure listing and the
    per-role matching section.

    ``guess`` is the displayed initial-guess phrase, which may differ from the
    true ``predicted`` by one same-category foil word (see
    ``_choose_foil_guess``). Grammar/template detection still uses the true
    ``predicted``; only the guess text and first-word mention reflect the foil.
    """
    if guess is None:
        guess = predicted
    words = predicted.split(" ")
    guess_words = guess.split(" ")
    template = _detect_template(words)
    form = _TEMPLATE_FORM[template]
    lines = []
    first_word = guess_words[0] if guess_words else ""
    lines.append(
        f"The phrase is probably something like \"{guess}\". "
        f"But that might be wrong and we should only use it to match grammar. "
        f"We have {len(displayed_words)} words. "
        f"The first word is probably \"{first_word}\". "
        f"So the grammar matches pattern {form}. "
        f"Let's use this grammar, forget this phrase completely, and find true words in phrase by scoring. "
        # "Let's see. "
    )
    return lines


def _render_sentence_structure(
    predicted: str,
    displayed_words: Optional[List[str]] = None,
    used_plain: Optional["set[str]"] = None,
) -> List[str]:
    """Return CoT lines for per-role matchings + summary."""
    words = predicted.split(" ")
    n = len(words)
    template = _detect_template(words)

    if displayed_words is None:
        displayed_words = list(words)
    if used_plain is None:
        used_plain = set()

    lines: List[str] = []

    phrase_words = list(displayed_words)
    for role, idx in _roles_for_template(template):
        if idx >= n:
            continue
        gold = words[idx]
        dword = displayed_words[idx] if idx < len(displayed_words) else gold
        phrase_words[idx] = gold
        phrase_so_far = " ".join(phrase_words[: idx + 1]) + "."
        lines.append("")
        lines.append(
            _render_role_matchings(
                role,
                dword,
                used_plain,
                position=idx + 1,
                phrase_so_far=phrase_so_far,
            )
        )

    lines.append("")
    lines.append(f"Thus the final answer is the phrase so far: {predicted}")
    return lines


def _derive_word_mapping(
    cipher_word: str,
    candidate_plain: str,
    current_mapping: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """Return NEW cipher->plain mappings implied by aligning ``cipher_word``
    with ``candidate_plain``, or ``None`` if the alignment is inconsistent.

    A candidate is inconsistent if it contradicts ``current_mapping``, violates
    injectivity (two cipher letters forced to the same plain letter), or
    collides a new mapping with a plain letter already in use.
    """
    if len(cipher_word) != len(candidate_plain):
        return None
    new: Dict[str, str] = {}
    used_plain = set(current_mapping.values())
    for c, p in zip(cipher_word, candidate_plain):
        if not c.isalpha():
            if c != p:
                return None
            continue
        if c in current_mapping:
            if current_mapping[c] != p:
                return None
        elif c in new:
            if new[c] != p:
                return None
        else:
            if p in used_plain or p in new.values():
                return None
            new[c] = p
    return new


def _fill_from_vocab(
    cipher_words: List[str],
    mapping: Dict[str, str],
) -> Dict[str, str]:
    """Iteratively extend ``mapping`` by matching each decrypted query word
    against ``_VOCAB``. For each word still containing unknowns, pick the
    highest-frequency vocab entry whose letter alignment is consistent with
    the current mapping, and commit the implied new letter mappings. Loops
    until no word admits a new inference.
    """
    resolved = dict(mapping)
    changed = True
    while changed:
        changed = False
        for cw in cipher_words:
            if "?" not in _decrypt_word(cw, resolved):
                continue
            best_new: Optional[Dict[str, str]] = None
            best_count = -1
            for cand, count in _VOCAB.items():
                if len(cand) != len(cw):
                    continue
                new = _derive_word_mapping(cw, cand, resolved)
                if new is None or not new:
                    continue
                if count > best_count:
                    best_count = count
                    best_new = new
            if best_new:
                resolved.update(best_new)
                changed = True
    return resolved


def _decrypt_word(word: str, mapping: Dict[str, str]) -> str:
    return "".join(mapping.get(c, "?") if c.isalpha() else c for c in word)


def _render_decryption(
    query_cipher: str,
    final_mapping: Dict[str, str],
    confusion_p: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str, Tuple[Optional[str], Optional[str]]]:
    """Render the decryption block.

    Returns ``(block_text, displayed_decryption, predicted, foil)``:

    * ``block_text`` — full multi-line decryption section including per-word
      lines and trailing summary lines.
    * ``displayed_decryption`` — the ``"..."`` string after "decrypted phrase
      is:" which may contain ``?`` for still-unknown cipher letters at the
      time of decryption.
    * ``predicted`` — best-effort resolved decryption (``?`` filled in via
      process-of-elimination when possible).
    * ``foil`` — ``(foil_word, role)`` if the displayed initial guess swapped
      one word for a same-category foil (see ``_choose_foil_guess``), else
      ``(None, None)``.
    """
    cipher_words = query_cipher.split(" ")

    resolved_mapping = _resolve_unknowns(final_mapping)
    resolved_mapping = _fill_from_vocab(cipher_words, resolved_mapping)
    resolved_mapping = _resolve_unknowns(resolved_mapping)

    used_plain = set(final_mapping.values())
    # Walk the cipher side a..z and list each mapped plain letter once, in
    # cipher-letter order (so the listing reads "the letter a maps to, then
    # the letter b maps to, ..."). Unmapped cipher letters are skipped.
    output_letters: List[str] = []
    seen: "set[str]" = set()
    first_arrow: Optional[Tuple[str, str]] = None
    for i in range(26):
        c = chr(ord("a") + i)
        v = final_mapping.get(c)
        if isinstance(v, str) and len(v) == 1 and v.isalpha() and v not in seen:
            seen.add(v)
            output_letters.append(v)
            if first_arrow is None:
                first_arrow = (c, v)
    lines: List[str] = []
    lines.extend(
        _render_output_letters_block(used_plain, output_letters, first_arrow)
    )
    lines.append("")

    lines.append(f"Now we have mapping for many letters. Let's decode target phrase: {query_cipher}")
    # lines.append(query_cipher)
    displayed_words: List[str] = []
    for i, cw in enumerate(cipher_words):
        pref = " "  #_prefix(i) override b/c target phrase has space before it
        lines.append(f"{pref}{cw}")
        plain_with_q = _decrypt_word(cw, final_mapping)
        for c, p in zip(cw, plain_with_q):
            lines.append(f"{c} -> {p}")
        lines.append(f"=> {_spaced(plain_with_q)}")
        displayed_words.append(plain_with_q)

    predicted = " ".join(
        _decrypt_word(w, resolved_mapping) for w in cipher_words
    )

    if rng is not None:
        guess, foil_word, foil_role = _choose_foil_guess(predicted, confusion_p, rng)
    else:
        guess, foil_word, foil_role = predicted, None, None

    lines.append("")
    lines.extend(_render_vocab_block())
    lines.append("")
    lines.extend(_render_structures_block())
    lines.append("")
    lines.extend(_render_match_block(displayed_words, predicted, used_plain, guess))
    lines.extend(
        _render_sentence_structure(predicted, displayed_words, used_plain)
    )

    displayed = " ".join(displayed_words)
    return "\n".join(lines), displayed, predicted, (foil_word, foil_role)


# ---------------------------------------------------------------------------
# CoT rendering
# ---------------------------------------------------------------------------

def _render_cot(
    examples: List[Tuple[str, str]],
    query_cipher: str,
    confusion_p: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, Tuple[Optional[str], Optional[str]]]:
    """Return ``(cot_text, predicted, foil)``."""
    lines: List[str] = [
        "We need to figure out substitution cipher mapping from examples. "
        "Let's parse examples.\n"
    ]

    cumulative: Dict[str, str] = {}
    for idx, (cipher_sentence, plain_sentence) in enumerate(examples, start=1):
        lines.append(f"Example {idx}")
        block, cumulative = _render_example_block(
            cipher_sentence, plain_sentence, cumulative
        )
        lines.append(block)
        lines.append("")
        lines.append(_render_mapping_block(cumulative))
        lines.append("")

    decrypt_block, _displayed, predicted, foil = _render_decryption(
        query_cipher, cumulative, confusion_p, rng,
    )
    lines.append(decrypt_block)

    return "\n".join(lines), predicted, foil


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cot(
    prompt: str,
    answer: str,
    *,
    confusion_p: float = 0.0,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Build a text_cipher CoT from the prompt.

    ``confusion_p`` (config: ``data.cot_kwargs.text_cipher.confusion_p``) is the
    per-phrase probability that the displayed initial guess swaps one word for a
    same-category foil; the per-role scoring still recovers the truth, so the
    final answer is unchanged. The foil RNG is seeded deterministically from the
    query so traces are reproducible across multiprocess workers.
    """
    examples = _parse_examples(prompt)
    query_cipher = _parse_query(prompt)

    confusion_p = float(confusion_p)
    seed = int.from_bytes(
        hashlib.md5(query_cipher.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed)

    cot_text, predicted, (foil_word, foil_role) = _render_cot(
        examples, query_cipher, confusion_p, rng,
    )

    correct: Optional[bool] = (predicted == answer.strip())
    return cot_text, {
        "predicted": predicted,
        "correct": correct,
        "n_examples": len(examples),
        "query_cipher": query_cipher,
        "confused": foil_word,
        "confused_role": foil_role,
    }
