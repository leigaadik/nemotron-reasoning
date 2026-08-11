"""Synthetic generator for the roman_numeral category.

CoT parser key regex
--------------------
Query extraction:  ``write the number\\s+(\\d+)``

Prompt format (must match real data)
-------------------------------------
::

    In Alice's Wonderland, numbers are secretly converted into a different
    numeral system. Some examples are given below:
    11 -> XI
    15 -> XV
    94 -> XCIV
    19 -> XIX
    Now, write the number 38 in the Wonderland numeral system.

Data generation (reverse-engineered from train.csv, 1576 rows)
--------------------------------------------------------------
* Target range: [1, 100] — confirmed min=1, max=100, all 100 integers present.
* Target distribution: uniform over [1, 100] (chi-square p=0.34, consistent
  with i.i.d. random.randint(1, 100)).
* n_examples per prompt: 3 (36.7%), 4 (31.7%), 5 (31.5%) — sample with those
  weights. Examples are drawn uniformly from [1, 100] \\ {target}.
* Example value distribution: uniform over [1, 100] (chi-square p=0.41).

Harder subtractive forms in [1, 100]
-------------------------------------
IV (4), IX (9), XL (40–49), XC (90–99).
XL and XC are the least intuitive; oversample 40–49 and 90–99 to bias
the generator toward these harder cases.

Oversampling weights
--------------------
* 44, 49, 94, 99: 4× (double subtractive: XL+IV, XL+IX, XC+IV, XC+IX)
* 40–49 (excl. 44, 49): 2× (XL subtractive prefix)
* 90–99 (excl. 94, 99): 2× (XC subtractive prefix)
* 100: 4×
* all others: 1×
"""

import random
from typing import Dict, List, Tuple

from nemotron.data.main._syn_base import SyntheticGenerator, make_id

_VALS: List[Tuple[int, str]] = [
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

_DOUBLE_SUBTRACTIVE = {44, 49, 94, 99}
_XL_RANGE = set(range(40, 50))
_XC_RANGE = set(range(90, 100))

_POOL: List[int] = []
_WEIGHTS: List[int] = []
for _i in range(1, 101):
    _POOL.append(_i)
    if _i in _DOUBLE_SUBTRACTIVE:
        _WEIGHTS.append(4)
    elif _i in _XL_RANGE or _i in _XC_RANGE:
        _WEIGHTS.append(2)
    elif _i == 100:
        _WEIGHTS.append(4)
    else:
        _WEIGHTS.append(1)

_N_EXAMPLES_POPULATION = [3, 4, 5]
_N_EXAMPLES_WEIGHTS = [0.367, 0.317, 0.315]

_HEADER = (
    "In Alice's Wonderland, numbers are secretly converted into a different "
    "numeral system. Some examples are given below:"
)
_FOOTER_TEMPLATE = "Now, write the number {target} in the Wonderland numeral system."


def _to_roman(n: int) -> str:
    result = ""
    for value, symbol in _VALS:
        while n >= value:
            result += symbol
            n -= value
    return result


class RomanNumeralGenerator(SyntheticGenerator):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)

    def generate_one(self) -> Dict[str, str]:
        rng = self._rng
        target = rng.choices(_POOL, weights=_WEIGHTS, k=1)[0]
        n = rng.choices(_N_EXAMPLES_POPULATION, weights=_N_EXAMPLES_WEIGHTS, k=1)[0]

        example_pool = [x for x in range(1, 101) if x != target]
        examples = rng.sample(example_pool, n)

        example_lines = [f"{x} -> {_to_roman(x)}" for x in examples]
        prompt = (
            _HEADER + "\n"
            + "\n".join(example_lines) + "\n"
            + _FOOTER_TEMPLATE.format(target=target)
        )
        return {"id": make_id(), "prompt": prompt, "answer": _to_roman(target)}
