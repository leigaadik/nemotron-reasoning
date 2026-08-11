"""Synthetic generator for the physics_gravity category.

CoT parser key regex
--------------------
Examples:  ``t\\s*=\\s*([\\d.]+)\\s*s,\\s*distance\\s*=\\s*([\\d.]+)\\s*m``
Query t:   ``falling distance for t\\s*=\\s*([\\d.]+)\\s*s``

Prompt format (must match real data)
-------------------------------------
::

    In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
    For t = 1.37s, distance = 14.92 m
    For t = 4.27s, distance = 144.96 m
    ...
    Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.

Implementation notes
--------------------
* g drawn from uniform [4.8, 20.0] (continuous float, not rounded).
* 3–5 example (t, d) pairs; t drawn from uniform [1.0, 5.0], rounded to
  2 decimal places, formatted via ``str(round(t, 2))`` (trailing zeros
  stripped by Python, e.g. 1.70 → '1.7', 2.00 → '2.0').
* d = 0.5 * g * t², formatted identically: ``str(round(d, 2))``.
* Query t is distinct from all example t values (same format).
* Answer = ``str(round(0.5 * g * t_query², 2))``.
"""

import random
from typing import Dict, List

from nemotron.data.main._syn_base import SyntheticGenerator, make_id


class PhysicsGravityGenerator(SyntheticGenerator):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._rng = random.Random(seed)

    def generate_one(self) -> Dict[str, str]:
        rng = self._rng

        g = rng.uniform(4.8, 20.0) # Found [4.9, 19.6], I guess [0.5 g, 2.0 g]
        n_examples = rng.randint(3, 5)

        # Sample n_examples + 1 distinct t values (for examples + query)
        t_values: List[float] = []
        seen: set = set()
        while len(t_values) < n_examples + 1:
            t_raw = rng.uniform(0.9, 5.5)  # Found [1, 5]
            t = round(t_raw, 2)
            if t not in seen:
                seen.add(t)
                t_values.append(t)

        example_ts = t_values[:n_examples]
        query_t = t_values[-1]

        lines: List[str] = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed."
            " Here are some example observations:"
        ]
        for t in example_ts:
            d = round(0.5 * g * t ** 2, 2)
            lines.append(f"For t = {t}s, distance = {d} m")

        lines.append(
            f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2."
        )

        answer = str(round(0.5 * g * query_t ** 2, 2))

        return {"id": make_id(), "prompt": "\n".join(lines), "answer": answer}
