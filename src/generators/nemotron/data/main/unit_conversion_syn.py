"""Synthetic generator for the unit_conversion category.

CoT parser key regex
--------------------
Examples:   ``([\\d.]+)\\s+\\w+\\s+becomes\\s+([\\d.]+)``
Query val:  ``convert the following measurement[:\\s]+([\\.\\d]+)``

Prompt format (must match real data)
-------------------------------------
::

    In Alice's Wonderland, a secret unit conversion is applied to
    measurements. For example:
    10.08 m becomes 6.69
    17.83 m becomes 11.83
    ...
    Now, convert the following measurement: 25.09 m

Implementation notes
--------------------
* ``k ~ Uniform(0.5, 2.0)`` — full float precision, no rounding.
* ``n_examples`` drawn uniformly from ``{3, 4, 5}``.
* Input values drawn from ``Uniform(5.0, 50.0)``, formatted with ``:.2f``.
* All outputs (example and answer) computed as ``round(k * x, 2)`` and
  formatted with ``:.2f`` to preserve trailing zeros.
"""

import random
from typing import Dict, List

from nemotron.data.main._syn_base import SyntheticGenerator, make_id

_HEADER = (
    "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
    "For example:"
)
_FOOTER_TEMPLATE = "Now, convert the following measurement: {query} m"


class UnitConversionGenerator(SyntheticGenerator):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)

    def generate_one(self) -> Dict[str, str]:
        rng = self._rng
        # 10% identity (k=1); otherwise k ~ Uniform(0.45, 2.2)
        if rng.random() < 0.10:
            k = 1.0
        else:
            k = rng.uniform(0.45, 2.2)  # True range: [0.5, 2.0]  # cover a bit more
        n = rng.choice([3, 4, 5])

        # Draw n+1 distinct input values with 2 decimal places
        seen = set()
        inputs: List[float] = []
        while len(inputs) < n + 1:
            x = round(rng.uniform(5.0, 50.0), 2)
            if x not in seen:
                seen.add(x)
                inputs.append(x)

        example_inputs = inputs[:n]
        query_val = inputs[n]

        example_lines = [
            f"{round(x, 2)} m becomes {round(k * x, 2):.2f}"
            for x in example_inputs
        ]
        answer = f"{round(k * query_val, 2):.2f}"

        prompt = (
            _HEADER + "\n"
            + "\n".join(example_lines)
            + "\n"
            + _FOOTER_TEMPLATE.format(query=round(query_val, 2))
        )
        return {"id": make_id(), "prompt": prompt, "answer": answer}
