"""Synthetic generator for the text_cipher category.

CoT parser key regex
--------------------
Examples: lines of the form ``<ciphertext_phrase> -> <plaintext_phrase>``
Query:    text after ``Now, decrypt the following text:``

Prompt format (must match real data)
-------------------------------------
::

    In Alice's Wonderland, secret encryption rules are used on text. Here
    are some examples:
    ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
    pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
    ...
    Now, decrypt the following text: hncreo learpaq qaleap

Implementation notes
--------------------
* Use ONLY words from the ``_VOCAB`` mapping in
  ``nemotron.data.main.text_cipher_cot`` — import it directly.
* Generate a random bijective 26→26 letter permutation (the cipher).
* Build 3–5 example sentences (3–5 words each from VOCAB); encrypt each.
* Generate a 3–5 word query phrase from VOCAB; encrypt it for the prompt.
* Answer = plaintext query phrase (space-joined words).
* Preamble must contain ``encryption rules`` for category auto-detection.

Sentence templates (reverse-engineered from train.csv; each ~25%)
------------------------------------------------------------------
1. S V O             — e.g. "dragon chases crystal"
2. S V PREP LOC      — e.g. "alice explores beyond forest"
3. the ADJ S V       — e.g. "the wise dragon watches"
4. S V the ADJ O     — e.g. "alice creates the magical crystal"

Word categories (all slots sampled uniformly)
---------------------------------------------
SUBJECT (15):  alice bird cat dragon hatter king knight mouse princess
               queen rabbit student teacher turtle wizard
VERB (14):     chases creates discovers draws dreams explores follows found
               imagines reads sees studies watches writes
OBJECT (15):   book castle crystal door forest garden key map message
               mirror potion puzzle secret story treasure
LOCATION (14): castle cave forest garden island library mountain ocean
               palace school tower valley village wonderland
PREP (8):      above around beyond in inside near through under
ADJECTIVE (14):ancient bright clever colorful curious dark golden hidden
               magical mysterious secret silver strange wise

Note: castle/forest/garden appear in both OBJECT and LOCATION;
      secret appears in both OBJECT and ADJECTIVE.
"""

import random
import string
from typing import Dict, List, Tuple

from nemotron.data.main._syn_base import SyntheticGenerator, make_id

# ---------------------------------------------------------------------------
# Word lists per slot (order does not matter; sampling is uniform)
# ---------------------------------------------------------------------------

_SUBJECT: List[str] = [
    "alice", "bird", "cat", "dragon", "hatter", "king", "knight", "mouse",
    "princess", "queen", "rabbit", "student", "teacher", "turtle", "wizard",
]

_VERB: List[str] = [
    "chases", "creates", "discovers", "draws", "dreams", "explores", "follows",
    "found", "imagines", "reads", "sees", "studies", "watches", "writes",
]

_OBJECT: List[str] = [
    "book", "castle", "crystal", "door", "forest", "garden", "key", "map",
    "message", "mirror", "potion", "puzzle", "secret", "story", "treasure",
]

_LOCATION: List[str] = [
    "castle", "cave", "forest", "garden", "island", "library", "mountain",
    "ocean", "palace", "school", "tower", "valley", "village", "wonderland",
]

_PREP: List[str] = [
    "above", "around", "beyond", "in", "inside", "near", "through", "under",
]

_ADJECTIVE: List[str] = [
    "ancient", "bright", "clever", "colorful", "curious", "dark", "golden",
    "hidden", "magical", "mysterious", "secret", "silver", "strange", "wise",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_cipher(rng: random.Random) -> Dict[str, str]:
    """Return a uniformly random bijective letter permutation (cipher map)."""
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _encrypt(phrase: str, cipher: Dict[str, str]) -> str:
    return "".join(cipher.get(ch, ch) for ch in phrase)


_ROLE_CATEGORIES: Dict[str, List[str]] = {
    "subject": _SUBJECT,
    "verb": _VERB,
    "object": _OBJECT,
    "location": _LOCATION,
    "preposition": _PREP,
    "adjective": _ADJECTIVE,
}


def _detect_template(words: List[str]) -> int:
    if not words:
        return 1
    if words[0] == "the":
        return 3
    if len(words) >= 3 and words[2] in _PREP:
        return 2
    if len(words) >= 3 and words[2] == "the":
        return 4
    return 1


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


def _build_partial_mapping(
    examples: List[Tuple[str, str]],
) -> Dict[str, str]:
    """First-assignment-wins cipher->plain map from (cipher, plain) pairs."""
    mapping: Dict[str, str] = {}
    for cipher_text, plain_text in examples:
        for c, p in zip(cipher_text, plain_text):
            if c.isalpha() and c not in mapping:
                mapping[c] = p
    return mapping


def _decrypt_word(cipher_word: str, mapping: Dict[str, str]) -> str:
    return "".join(
        mapping.get(c, "?") if c.isalpha() else c for c in cipher_word
    )


def _has_max_tie(
    query_plain: str,
    query_cipher: str,
    mapping: Dict[str, str],
) -> bool:
    """True if any role in the query has 2+ candidates tied at max
    letter-match count under the (length within 1) filter — i.e. the
    matchings step would require deliberation."""
    plain_words = query_plain.split(" ")
    cipher_words = query_cipher.split(" ")
    if len(plain_words) != len(cipher_words):
        return False
    template = _detect_template(plain_words)
    for role, idx in _roles_for_template(template):
        if idx >= len(plain_words) or role == "the":
            continue
        cat = _ROLE_CATEGORIES.get(role)
        if cat is None:
            continue
        dword = _decrypt_word(cipher_words[idx], mapping)
        max_count = -1
        n_max = 0
        for cand in cat:
            if abs(len(dword) - len(cand)) > 1:
                continue
            min_len = min(len(dword), len(cand))
            count = sum(
                1 for i in range(min_len)
                if dword[i] != "?" and dword[i] == cand[i]
            )
            if count > max_count:
                max_count = count
                n_max = 1
            elif count == max_count:
                n_max += 1
        if n_max >= 2:
            return True
    return False


def _weighted_choice(
    rng: random.Random, pool: List[str], weights: Dict[str, float]
) -> str:
    """Uniform pick unless *weights* overrides specific words (default 1.0).

    A word with weight ``w`` is picked ``w`` times as often as a weight-1 word
    in the same pool, so ``w = 5`` means ~5x the per-word query frequency.
    """
    if not weights:
        return rng.choice(pool)
    return rng.choices(pool, weights=[weights.get(x, 1.0) for x in pool], k=1)[0]


def _sample_sentence(
    rng: random.Random, weights: Dict[str, float] = None
) -> str:
    """Sample one sentence from the four templates with equal probability.

    Content-word slots use *weights* (per-word query upsampling); the literal
    "the" is fixed. Only the query is weighted — example sentences come from
    ``_cover_examples`` and stay uniform.
    """
    template = rng.randint(1, 4)
    if template == 1:
        # S V O
        return " ".join([
            _weighted_choice(rng, _SUBJECT, weights),
            _weighted_choice(rng, _VERB, weights),
            _weighted_choice(rng, _OBJECT, weights),
        ])
    elif template == 2:
        # S V PREP LOC
        return " ".join([
            _weighted_choice(rng, _SUBJECT, weights),
            _weighted_choice(rng, _VERB, weights),
            _weighted_choice(rng, _PREP, weights),
            _weighted_choice(rng, _LOCATION, weights),
        ])
    elif template == 3:
        # the ADJ S V
        return " ".join([
            "the",
            _weighted_choice(rng, _ADJECTIVE, weights),
            _weighted_choice(rng, _SUBJECT, weights),
            _weighted_choice(rng, _VERB, weights),
        ])
    else:
        # S V the ADJ O
        return " ".join([
            _weighted_choice(rng, _SUBJECT, weights),
            _weighted_choice(rng, _VERB, weights),
            "the",
            _weighted_choice(rng, _ADJECTIVE, weights),
            _weighted_choice(rng, _OBJECT, weights),
        ])


# ---------------------------------------------------------------------------
# Difficulty features (bucket = max #missing letters in any single word)
# ---------------------------------------------------------------------------
#
# Key fact exploited by the constructive builder below: a query letter is
# "under-determined" (unmapped) iff that *plaintext* letter never appears in any
# example *plaintext*. The cipher is a bijection, so a ciphertext letter is
# absent from the examples exactly when its plaintext preimage is. Hiding a set
# of plaintext letters H from the examples therefore makes exactly the query
# positions bearing those letters unrecoverable — independent of the random
# cipher. This turns every difficulty knob into a construction parameter rather
# than a rejection-sampling gamble.

_NUM_BUCKETS: int = 5  # max_missing in {0, 1, 2, 3, >=4}; bucket >=4 capped at 4


def _word_missing_count(cipher_word: str, mapping: Dict[str, str]) -> int:
    return _decrypt_word(cipher_word, mapping).count("?")


def _bucket(q_cipher: str, mapping: Dict[str, str]) -> int:
    """Bucket index 0..4; 4 means '>= 4 missing letters in some word'."""
    counts = [_word_missing_count(w, mapping) for w in q_cipher.split(" ")]
    return min(max(counts) if counts else 0, 4)


def _first_letter_missing(q_cipher: str, mapping: Dict[str, str]) -> bool:
    for w in q_cipher.split(" "):
        d = _decrypt_word(w, mapping)
        if d and d[0] == "?":
            return True
    return False


def _last_word_missing(q_cipher: str, mapping: Dict[str, str]) -> bool:
    words = q_cipher.split(" ")
    return bool(words) and "?" in _decrypt_word(words[-1], mapping)


def _once_letters(word: str) -> List[str]:
    """Letters occurring exactly once in *word* (hiding one ⇒ exactly one ?)."""
    counts: Dict[str, int] = {}
    for c in word:
        counts[c] = counts.get(c, 0) + 1
    return [c for c in word if counts[c] == 1]


# Literal-word slot layout per template, matching ``_sample_sentence`` /
# ``_roles_for_template``. ``"the"`` is a fixed token (no role pool).
_TEMPLATE_SLOTS: Dict[int, List[str]] = {
    1: ["subject", "verb", "object"],
    2: ["subject", "verb", "preposition", "location"],
    3: ["the", "adjective", "subject", "verb"],
    4: ["subject", "verb", "the", "adjective", "object"],
}

_THE_LETTERS = frozenset("the")


def _unique_fit(decoded_word: str, role: str) -> bool:
    """True iff exactly one word in *role*'s pool matches the visible letters
    (``?`` = wildcard) and length of *decoded_word* — i.e. the under-determined
    word is still uniquely recoverable, so the reference CoT label is sound."""
    cat = _ROLE_CATEGORIES.get(role)
    if cat is None:
        return False
    fits = [
        w for w in cat
        if len(w) == len(decoded_word)
        and all(d == "?" or d == c for d, c in zip(decoded_word, w))
    ]
    return len(fits) == 1


def _covering_sentence(
    rng: random.Random,
    want: set,
    hidden: set,
    max_attempts: int = 40,
) -> str:
    """One example sentence: every word avoids all letters in *hidden*, chosen
    greedily to cover as many letters of *want* as possible. Returns None if no
    template can be filled (some role pool is empty under *hidden*)."""
    # Templates carrying the literal "the" are unusable when t/h/e is hidden.
    allowed = [1, 2] if (hidden & _THE_LETTERS) else [1, 2, 3, 4]
    for _ in range(max_attempts):
        template = rng.choice(allowed)
        words: List[str] = []
        local: set = set()
        ok = True
        for role in _TEMPLATE_SLOTS[template]:
            if role == "the":
                words.append("the")
                continue
            pool = [w for w in _ROLE_CATEGORIES[role] if not (set(w) & hidden)]
            if not pool:
                ok = False
                break
            target = want - local
            best = max(pool, key=lambda w: (len(set(w) & target), rng.random()))
            words.append(best)
            local |= set(best)
        if ok:
            return " ".join(words)
    return None


def _cover_examples(
    rng: random.Random,
    query_letters: set,
    hidden: set,
    n_examples: int,
) -> List[str]:
    """Build *n_examples* sentences covering ``query_letters - hidden`` while
    containing none of *hidden*. Returns None if coverage isn't achieved."""
    needed = set(query_letters) - hidden
    covered: set = set()
    examples: List[str] = []
    for _ in range(n_examples):
        s = _covering_sentence(rng, needed - covered, hidden)
        if s is None:
            return None
        examples.append(s)
        covered |= set(s.replace(" ", ""))
    if not needed <= covered:
        return None
    return examples


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_QUERY_RESAMPLES: int = 20


class TextCipherGenerator(SyntheticGenerator):
    def __init__(
        self,
        seed: int = 42,
        difficulty_buckets: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.30, 0.10),
        first_letter_floor: float = 0.40,
        last_word_floor: float = 0.40,
        host_min_len: int = 3,
        the_hidden_p: float = 0.0,
        word_weights: Dict[str, float] = None,
        max_build_tries: int = 80,
        deliberation_oversample_prob: float = None,  # DEPRECATED, ignored
        max_resample_attempts: int = None,           # DEPRECATED, ignored
    ) -> None:
        """Difficulty-controlled text_cipher generation.

        The generator is *stateful across a ``generate(n)`` run*: it tracks how
        many samples it has emitted per difficulty bucket and how many satisfy
        the two marginal floors, and steers each new sample to whichever target
        is most behind — so the realized distribution tracks the targets exactly
        rather than only in expectation.

        Parameters
        ----------
        difficulty_buckets:
            Target fractions for bucket = max #under-determined letters in any
            single query word, indexed ``0,1,2,3,>=4``. Must have 5 entries.
        first_letter_floor:
            Minimum fraction of queries with a word whose decode starts with a
            ``?`` (word-initial under-determined letter — the dominant failure
            mode). Enforced as a floor via "require when running below".
        last_word_floor:
            Minimum fraction of queries whose LAST word is under-determined
            (training proxy for the observed 4.45x last-word error hotspot).
        host_min_len:
            Minimum length of the host word used for buckets >= 2. Default 3
            (effectively a no-op — every content word is >= 3 chars), so short
            words can host in the harder buckets too. Label soundness no longer
            relies on this: ``_unique_fit`` is the sole guard and rejects any
            build where the under-determined word isn't uniquely recoverable, so
            e.g. "key" still can't reach bucket 3 ("???" ties with "map").
        word_weights:
            Per-word query-frequency multipliers (default 1.0 for any word not
            listed). A word with weight ``w`` appears in the query ~``w`` times
            as often as a weight-1 word in the same slot pool. Only the query is
            weighted; example sentences stay uniform. Used to upsample the top
            content-word confusions (e.g. bird/wizard, book/door/key). None or
            {} = uniform (default; behavior unchanged).
        the_hidden_p:
            Fraction of queries that deliberately under-determine a letter of
            the literal article "the" (templates 3/4), so the decode shows e.g.
            "t?e"/"?he"/"th?". Targets the observed eval failure where a hidden
            "the" is misread as a subject (e.g. "teacher"), flipping the model to
            the wrong template. 0.0 disables it (default; behavior unchanged).
            The literal "the" is uniquely recoverable from its template slot, so
            these stay solvable; the reference CoT already decodes them correctly.
        max_build_tries:
            Inner attempt cap per constructive build before the controller
            relaxes constraints (drop floors, then downgrade the bucket).
        deliberation_oversample_prob, max_resample_attempts:
            DEPRECATED no-ops, retained so existing configs/imports don't break.
            The old max-tie oversampler selected for genuine ambiguity, which
            appears in 0% of real eval failures (see scripts/discovery/
            text_cipher_phase0.py); difficulty_buckets replaces it.
        """
        super().__init__(seed)
        self._rng = random.Random(seed)
        if len(difficulty_buckets) != _NUM_BUCKETS:
            raise ValueError(
                f"difficulty_buckets must have {_NUM_BUCKETS} entries, "
                f"got {len(difficulty_buckets)}"
            )
        total = sum(difficulty_buckets)
        self._bucket_targets = tuple(b / total for b in difficulty_buckets)
        self._first_floor = first_letter_floor
        self._last_floor = last_word_floor
        self._host_min_len = host_min_len
        self._the_hidden_p = the_hidden_p
        self._word_weights = word_weights or {}
        self._max_build_tries = max_build_tries
        # running state (across generate(n))
        self._counts = [0] * _NUM_BUCKETS
        self._first_count = 0
        self._last_count = 0
        self._n = 0
        self._fallbacks = 0  # times the controller had to relax a constraint
        self._the_hidden_n = 0  # queries emitted with an under-determined "the"

    # -- constructive build -------------------------------------------------

    def _build_once(
        self,
        target_k: int,
        need_first: bool,
        need_last: bool,
    ) -> Tuple[str, str, List[str], int, bool, bool]:
        """Try ONE constructive build for the requested bucket/floors.

        Returns ``(q_plain, q_cipher, example_plain_lines, bucket, first, last)``
        or None if this attempt failed (caller retries / relaxes).
        """
        rng = self._rng
        q_plain = _sample_sentence(rng, self._word_weights)
        words = q_plain.split(" ")
        n = len(words)
        template = _detect_template(words)
        role_of = {idx: role for role, idx in _roles_for_template(template)}

        # choose the host word (where the target_k missing letters live)
        hidden: set = set()
        if target_k > 0:
            forbidden = _THE_LETTERS if "the" in words else frozenset()
            cand = []
            for i, w in enumerate(words):
                role = role_of.get(i)
                if role is None or role == "the" or role == "preposition":
                    continue
                if target_k >= 2 and len(w) < self._host_min_len:
                    continue
                once = [c for c in _once_letters(w) if c not in forbidden]
                if need_first:
                    if w[0] in forbidden or w[0] not in once:
                        continue
                if len(once) >= target_k:
                    cand.append((i, once))
            if need_last:
                cand = [(i, o) for i, o in cand if i == n - 1]
            if not cand:
                return None
            host_idx, once = rng.choice(cand)
            chosen: set = set()
            if need_first:
                chosen.add(words[host_idx][0])
            pool = [c for c in once if c not in chosen]
            rng.shuffle(pool)
            for c in pool:
                if len(chosen) >= target_k:
                    break
                chosen.add(c)
            hidden = chosen

        query_letters = set("".join(words))
        n_examples = rng.randint(3, 5)
        examples = _cover_examples(rng, query_letters, hidden, n_examples)
        if examples is None:
            return None

        cipher = _random_cipher(rng)
        ex_cipher = [_encrypt(p, cipher) for p in examples]
        q_cipher = _encrypt(q_plain, cipher)
        mapping = _build_partial_mapping(list(zip(ex_cipher, examples)))

        # validate the realized features against the request
        bucket = _bucket(q_cipher, mapping)
        if bucket != target_k:
            return None
        first = _first_letter_missing(q_cipher, mapping)
        last = _last_word_missing(q_cipher, mapping)
        if need_first and not first:
            return None
        if need_last and not last:
            return None
        # solvability: every under-determined word is a content word with a
        # unique vocab fit; the literal "the" must stay fully resolved.
        cwords = q_cipher.split(" ")
        for i, cw in enumerate(cwords):
            d = _decrypt_word(cw, mapping)
            if "?" in d and not _unique_fit(d, role_of.get(i, "")):
                return None

        example_lines = [f"{ec} -> {ep}" for ec, ep in zip(ex_cipher, examples)]
        return q_plain, q_cipher, example_lines, bucket, first, last

    def _try_build(self, target_k, need_first, need_last):
        for _ in range(self._max_build_tries):
            r = self._build_once(target_k, need_first, need_last)
            if r is not None:
                return r
        return None

    def _build_the_hidden(self):
        """One build whose literal "the" is under-determined (a letter of t/h/e
        hidden), targeting the misread-"the"-as-subject failure. Parallel to
        ``_build_once`` but: it only uses templates that carry "the" (3/4), seeds
        ``hidden`` with one "the" letter, and treats the "the" slot as always
        solvable (the article is uniquely fixed by its template position). Counts
        toward whatever bucket/floors it realizes. Returns the same tuple as
        ``_build_once`` or None."""
        rng = self._rng
        for _ in range(self._max_build_tries):
            q_plain = _sample_sentence(rng, self._word_weights)
            words = q_plain.split(" ")
            if "the" not in words:
                continue
            template = _detect_template(words)
            role_of = {idx: role for role, idx in _roles_for_template(template)}

            # Hide one letter of "the" (uniform over t/h/e). All three positions
            # are recoverable from the fixed article; 'h' reproduces the observed
            # "t?e" case most directly but training all three generalizes.
            hidden = {rng.choice(list(_THE_LETTERS))}
            query_letters = set("".join(words))
            n_examples = rng.randint(3, 5)
            examples = _cover_examples(rng, query_letters, hidden, n_examples)
            if examples is None:
                continue

            cipher = _random_cipher(rng)
            ex_cipher = [_encrypt(p, cipher) for p in examples]
            q_cipher = _encrypt(q_plain, cipher)
            mapping = _build_partial_mapping(list(zip(ex_cipher, examples)))

            # The hidden "the" letter must actually surface in the query "the".
            the_idx = words.index("the")
            if "?" not in _decrypt_word(q_cipher.split(" ")[the_idx], mapping):
                continue

            # Solvability: content words keep a unique vocab fit; the literal
            # "the" is always recoverable from its slot, so it's exempt.
            cwords = q_cipher.split(" ")
            ok = True
            for i, cw in enumerate(cwords):
                d = _decrypt_word(cw, mapping)
                if "?" in d and role_of.get(i) != "the" and not _unique_fit(
                        d, role_of.get(i, "")):
                    ok = False
                    break
            if not ok:
                continue

            bucket = _bucket(q_cipher, mapping)
            first = _first_letter_missing(q_cipher, mapping)
            last = _last_word_missing(q_cipher, mapping)
            example_lines = [f"{ec} -> {ep}" for ec, ep in zip(ex_cipher, examples)]
            return q_plain, q_cipher, example_lines, bucket, first, last
        return None

    # -- quota controller ---------------------------------------------------

    def generate_one(self) -> Dict[str, str]:
        n = self._n
        # Targeted hidden-"the" injection (off when the_hidden_p == 0). Steered by
        # running deficit so the realized fraction tracks the_hidden_p; the sample
        # still counts toward whatever difficulty bucket / floors it lands in.
        if (self._the_hidden_p > 0.0
                and self._the_hidden_n < self._the_hidden_p * (n + 1)):
            built = self._build_the_hidden()
            if built is not None:
                self._the_hidden_n += 1
                return self._finalize(built)

        # bucket with the largest running deficit keeps the histogram exact
        deficits = [
            self._bucket_targets[b] * (n + 1) - self._counts[b]
            for b in range(_NUM_BUCKETS)
        ]
        target = max(range(_NUM_BUCKETS), key=lambda b: deficits[b])
        # floors only achievable for buckets that have missing letters
        need_first = target > 0 and self._first_count < self._first_floor * (n + 1)
        need_last = target > 0 and self._last_count < self._last_floor * (n + 1)

        built = self._try_build(target, need_first, need_last)
        if built is None and (need_first or need_last):
            # relax floors before touching the bucket distribution
            for nf, nl in ((need_first, False), (False, need_last), (False, False)):
                built = self._try_build(target, nf, nl)
                if built is not None:
                    self._fallbacks += 1
                    break
        if built is None:
            # last resort: downgrade the bucket (prefer adjacent easier ones)
            for t in sorted(range(_NUM_BUCKETS), key=lambda b: abs(b - target)):
                built = self._try_build(t, False, False)
                if built is not None:
                    self._fallbacks += 1
                    break

        return self._finalize(built)

    def _finalize(self, built) -> Dict[str, str]:
        """Update running state and render the problem dict (shared by the
        bucket-controller and hidden-"the" paths)."""
        q_plain, q_cipher, example_lines, bucket, first, last = built
        self._counts[bucket] += 1
        self._first_count += int(first)
        self._last_count += int(last)
        self._n += 1

        prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text."
            " Here are some examples:\n"
            + "\n".join(example_lines)
            + f"\nNow, decrypt the following text: {q_cipher}"
        )
        return {
            "id": make_id(),
            "prompt": prompt,
            "answer": q_plain,
        }
