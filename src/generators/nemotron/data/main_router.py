"""Main schema — top-level (6) + resolved (7) categories, CoT/syn registries, router.

Two parallel registries keyed by category name:

- ``COT_GENERATORS``  — modules exposing
  ``generate_cot(prompt, answer, **kwargs) -> (cot_text, meta)``
- ``SYN_GENERATORS``  — subclasses of ``SyntheticGenerator``

Top-level (6) categories are derived from prompt keywords. ``equation_transform``
splits into ``equation_transform/numeric`` vs ``equation_transform/symbol`` based
on whether the answer contains any digit — yielding 7 resolved keys that match
the registry keys above.

CoT generator contract
----------------------
Every ``generate_cot`` MUST take ``(prompt: str, answer: str, **kwargs)`` and
return a 2-tuple ``(cot_text: str, meta: Dict[str, Any])``.

``cot_text`` is the reasoning BODY only. It MUST NOT emit ``\\boxed{...}`` or
``</think>`` — the wrapper in :func:`nemotron.prompts.format_assistant` wraps
the body as ``<think>\\n{body}\\n</think>\\n\\boxed{answer}``. Prose references
to the answer inside the body (e.g. ``So the answer is X.``) are fine; they
just must not use ``\\boxed``.

``meta`` MUST contain:

- ``predicted``: Optional[str] — the generator's GENUINE derivation from the
  prompt. MUST be ``None`` when the generator could not actually derive an
  answer (fallback / guess branches). NEVER populate ``predicted`` from
  ``answer`` — doing so silently inflates the CoT-correctness metric
  computed in ``nemotron/data/gen_traces.py``.
- ``correct``: Optional[bool] — ``predicted == answer.strip()`` when
  ``predicted`` is not ``None``; ``None`` otherwise.

Generators may add extra diagnostic keys, but downstream code should only rely
on ``predicted`` and ``correct``.

Add a new category by importing its ``*_cot.py`` / ``*_syn.py`` module and
registering it in BOTH tables below (the module-load assertion will fail if
they drift).
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from nemotron.data.main import bit_manipulation_cot as _bit_manipulation_cot
from nemotron.data.main import equation_transform_numeric_cot as _eq_numeric_cot
from nemotron.data.main import equation_transform_symbol_cot as _eq_symbol_cot
from nemotron.data.main import physics_gravity_cot as _physics_gravity_cot
from nemotron.data.main import roman_numeral_cot as _roman_numeral_cot
from nemotron.data.main import text_cipher_cot as _text_cipher_cot
from nemotron.data.main import unit_conversion_cot as _unit_conversion_cot
from nemotron.data.main._syn_base import SyntheticGenerator
from nemotron.data.main.bit_manipulation_syn import BitManipulationGenerator
from nemotron.data.main.equation_transform_numeric_syn import (
    EquationTransformNumericGenerator,
)
from nemotron.data.main.equation_transform_symbol_syn import (
    EquationTransformSymbolGenerator,
)
from nemotron.data.main.physics_gravity_syn import PhysicsGravityGenerator
from nemotron.data.main.roman_numeral_syn import RomanNumeralGenerator
from nemotron.data.main.text_cipher_syn import TextCipherGenerator
from nemotron.data.main.unit_conversion_syn import UnitConversionGenerator


# ── Top-level (6) — source of truth ────────────────────────────────────────

_CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    ("bit_manipulation",      "bit manipulation rule"),
    ("physics_gravity",       "gravitational constant"),
    ("unit_conversion",       "unit conversion"),
    ("text_cipher",           "secret encryption rules are used on text"),
    ("roman_numeral",         "Wonderland numeral system"),
    ("equation_transform",    "a secret set of transformation rules is applied to equations"),
]
CATEGORIES: List[str] = [c for c, _ in _CATEGORY_KEYWORDS]


# ── Resolved (7) — source of truth ─────────────────────────────────────────

COT_GENERATORS = {
    "text_cipher":                  _text_cipher_cot,
    "roman_numeral":                _roman_numeral_cot,
    "unit_conversion":              _unit_conversion_cot,
    "physics_gravity":              _physics_gravity_cot,
    "bit_manipulation":             _bit_manipulation_cot,
    "equation_transform/numeric":   _eq_numeric_cot,
    "equation_transform/symbol":    _eq_symbol_cot,
}

SYN_GENERATORS: Dict[str, Type[SyntheticGenerator]] = {
    "text_cipher":                  TextCipherGenerator,
    "roman_numeral":                RomanNumeralGenerator,
    "unit_conversion":              UnitConversionGenerator,
    "physics_gravity":              PhysicsGravityGenerator,
    "bit_manipulation":             BitManipulationGenerator,
    "equation_transform/numeric":   EquationTransformNumericGenerator,
    "equation_transform/symbol":    EquationTransformSymbolGenerator,
}

# Categories whose assistant turn is rendered WITHOUT a ``<think>`` block and
# WITHOUT ``\boxed{...}``. Consumed by ``nemotron.prompts.format_assistant``
# (via ``dataset.format_chat``) and by ``gen_traces`` for both the assistant
# turn used in accuracy scoring and the chat-template ``enable_thinking``
# kwarg. The mask in ``gen_traces._make_target_mask`` falls back to the
# ``<|im_start|>assistant\n`` boundary when ``<think>\n`` is absent, so these
# rows still train end-to-end.
NO_THINKING_CATEGORIES: Set[str] = {}


def thinking_enabled(category: str) -> bool:
    """Return True iff this category's assistant turn should be wrapped in
    ``<think>...</think>\\n\\boxed{...}``. False categories render as a bare
    ``The final answer is: X`` line — see ``NO_THINKING_CATEGORIES``.
    """
    return category not in NO_THINKING_CATEGORIES


ID_TO_CAT: List[str] = list(COT_GENERATORS.keys())
CAT_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(ID_TO_CAT)}

assert set(COT_GENERATORS) == set(SYN_GENERATORS) == set(ID_TO_CAT), (
    "main schema drift: COT_GENERATORS, SYN_GENERATORS, and ID_TO_CAT must "
    "agree on keys"
)


# ── Detection ──────────────────────────────────────────────────────────────

def detect_category(prompt: str) -> str:
    for cat, kw in _CATEGORY_KEYWORDS:
        if kw in prompt:
            return cat
    return "unknown"


def resolve_category(row: Dict) -> str:
    """Return the canonical CoT-generator key for *row*.

    Top-level Kaggle categories pass through unchanged. ``equation_transform``
    is split into ``equation_transform/numeric`` vs ``equation_transform/symbol``
    by whether the answer contains any digit (covers plain ``6125``, signed
    ``-17``, and op-signed ``$46``/``56]``/``17/``/``{17`` encodings; pure
    symbol answers like ``@&``/``{}``/``'"[`` go to ``/symbol``).

    Used for both grouping/stats and CoT generator dispatch.
    """
    cat = row.get("category") or detect_category(row["prompt"])
    if cat == "equation_transform":
        answer = row.get("answer", "")
        if any(ch.isdigit() for ch in answer):
            return "equation_transform/numeric"
        return "equation_transform/symbol"
    return cat


# ── Dispatch ───────────────────────────────────────────────────────────────

def has_cot_generator(category: str) -> bool:
    """Return True if a CoT generator exists for this category."""
    return category in COT_GENERATORS or category == "equation_transform"


def generate_cot(row: Dict, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Dispatch to the appropriate CoT generator for the row.

    Returns ``(cot_text, meta)``. ``meta["predicted"]`` is the generator's
    genuine derivation from the prompt, or ``None`` on fallback branches — see
    the module docstring for the full contract.

    Raises KeyError if no generator is registered for the resolved category.

    Generators receive ``prompt``/``answer`` strings plus any forwarded
    ``**kwargs``; individual generators pick off the kwargs they understand.
    """
    cat = resolve_category(row)
    if cat not in COT_GENERATORS:
        raise KeyError(
            f"No CoT generator registered for category '{cat}'. "
            f"Available: {sorted(COT_GENERATORS)}"
        )
    gen_fn = COT_GENERATORS[cat].generate_cot
    prompt = row["prompt"]
    answer = row.get("answer", "") or ""
    return gen_fn(prompt, answer, **kwargs)


def get_generator(
    category: str,
    seed: int = 42,
    **kwargs,
) -> SyntheticGenerator:
    """Return an instantiated synthetic generator for *category*.

    Parameters
    ----------
    category : str
        One of the keys in :data:`SYN_GENERATORS`.
    seed : int
        RNG seed passed to the generator constructor.
    **kwargs
        Additional generator-specific kwargs (e.g. ``rule_mix`` for
        equation_transform_numeric). Unknown kwargs are silently ignored
        by generators that don't accept them.

    Raises
    ------
    KeyError
        If *category* is not registered.
    """
    if category not in SYN_GENERATORS:
        raise KeyError(
            f"No synthetic generator registered for category '{category}'. "
            f"Available: {sorted(SYN_GENERATORS)}"
        )
    cls = SYN_GENERATORS[category]
    sig = inspect.signature(cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(seed=seed, **accepted)


# ── Router instance facade ─────────────────────────────────────────────────

class MainRouter:
    name = "main"
    categories: List[str] = list(CATEGORIES)

    @property
    def subcategories(self) -> List[str]:
        """Subcategory CoT-generator keys (e.g. ``equation_transform/numeric``).

        These are accepted as ``--category`` filters in trace generation;
        rows are matched by :func:`resolve_category` rather than their raw
        ``category`` column.
        """
        return [k for k in COT_GENERATORS if "/" in k]

    def detect(self, prompt: str) -> str:
        return detect_category(prompt)

    def resolve(self, row: Dict) -> str:
        return resolve_category(row)

    def has_cot(self, category: str) -> bool:
        return has_cot_generator(category)

    def thinking_enabled(self, category: str) -> bool:
        return thinking_enabled(category)

    def generate_cot(self, row: Dict, **kwargs) -> Tuple[str, Dict[str, Any]]:
        return generate_cot(row, **kwargs)

    def reload_cot(self, category: str) -> str:
        # Reload mutates the module in place; COT_GENERATORS holds the same
        # module reference, so the next dispatch picks up the edit.
        # Helpers shared across files won't reload — restart for those.
        mod = COT_GENERATORS.get(category)
        if mod is None:
            raise KeyError(f"no CoT generator registered for category {category!r}")
        importlib.reload(mod)
        return mod.__name__

    def syn_generators(self) -> Dict:
        return dict(SYN_GENERATORS)


def get_router(name: str):
    if name == "main":
        return MainRouter()
    raise ValueError(f"unknown category schema: {name!r} (valid: main)")
