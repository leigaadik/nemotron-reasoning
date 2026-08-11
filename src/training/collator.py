"""Padding collator for pre-tokenized causal-LM records with explicit labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExplicitLabelsCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            size = len(feature["input_ids"])
            padding = max_length - size
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature.get("attention_mask", [1] * size) + [0] * padding)
            batch["labels"].append(feature["labels"] + [self.label_pad_token_id] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
