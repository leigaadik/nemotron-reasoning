"""Approximate stratified training order used by the reference notebook."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence


def build_stratified_index_order(
    labels: Sequence[str],
    batch_size: int,
    seed: int,
) -> list[int]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not labels:
        return []

    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[str(label)].append(idx)

    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)

    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    assigned = 0
    for label in sorted(by_label):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order
