"""Abstract base class for synthetic data generators.

Each category-specific generator inherits from ``SyntheticGenerator`` and
implements ``generate_one()``, which returns a single training example as a
dict with keys ``id``, ``prompt``, and ``answer``.

The ``prompt`` must use the exact same surface format as real training data so
that the existing CoT generators in ``nemotron.data.main`` can parse
it without modification.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List


class SyntheticGenerator(ABC):
    """Base class for category-specific synthetic prompt generators.

    Parameters
    ----------
    seed : int
        Initial RNG seed.  Subclasses should accept this in ``__init__`` and
        seed their random state accordingly so generation is reproducible.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @abstractmethod
    def generate_one(self) -> Dict[str, str]:
        """Return one synthetic training example.

        Returns
        -------
        Dict[str, str]
            A dict with the three required keys ``id``, ``prompt``,
            ``answer``:

            ``id``
                A unique string identifier (use :func:`_make_id`).
            ``prompt``
                The full problem statement.  Must be parseable by the
                corresponding CoT generator's regex patterns.
            ``answer``
                The ground-truth answer string, formatted exactly as it would
                appear in the real ``train.csv`` (e.g. ``"10101010"``,
                ``"XVI"``, ``"king chases castle"``).
        """

    def generate(self, n: int) -> List[Dict[str, str]]:
        """Generate *n* examples by calling :meth:`generate_one` repeatedly.

        Parameters
        ----------
        n : int
            Number of examples to generate.

        Returns
        -------
        List[Dict[str, str]]
            List of example dicts (``id``/``prompt``/``answer``).
        """
        return [self.generate_one() for _ in range(n)]


def make_id() -> str:
    """Return a UUID4 string usable as a row ``id``.

    Using UUID4 avoids collisions with the hex IDs in the real ``train.csv``
    (which are 8-character hex strings).
    """
    return str(uuid.uuid4())
