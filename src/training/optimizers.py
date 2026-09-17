"""Project-level optimizer configuration for Transformers Trainer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


SUPPORTED_OPTIMIZERS = (
    "adam",
    "adam_bnb_8bit",
    "adamw_torch",
    "adamw_torch_fused",
    "adamw_bnb_8bit",
    "sgd",
    "sgd_momentum",
    "muon",
)

_COMMON_KWARGS = {"lr", "weight_decay"}
_ADAM_KWARGS = {"betas", "eps"}


def _parse_optimizer_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("optimizer config must be a mapping")
    name = str(config.get("name", "")).lower()
    if name not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"Unsupported optimizer {name!r}; expected one of: {supported}")
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise TypeError("optimizer.kwargs must be a mapping")
    kwargs = dict(kwargs)

    duplicate = sorted(_COMMON_KWARGS.intersection(kwargs))
    if name.startswith("adam"):
        duplicate.extend(sorted(_ADAM_KWARGS.intersection(kwargs)))
    if duplicate:
        fields = ", ".join(duplicate)
        raise ValueError(
            f"Configure {fields} under training, not optimizer.kwargs, "
            "to keep a single source of truth"
        )
    return name, kwargs


def _validate_muon_parameters(model: Any) -> Counter[int]:
    dimensions = Counter(
        parameter.ndim for parameter in model.parameters() if parameter.requires_grad
    )
    invalid = [
        (name, tuple(parameter.shape))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.ndim != 2
    ]
    if invalid:
        examples = ", ".join(f"{name}: {shape}" for name, shape in invalid[:10])
        raise ValueError(
            "Pure Muon requires every trainable parameter to be 2D; "
            f"found {len(invalid)} incompatible parameter(s): {examples}"
        )
    if not dimensions:
        raise ValueError("Cannot create Muon optimizer: model has no trainable parameters")
    return dimensions


def build_optimizer_spec(
    config: Mapping[str, Any],
    training_args: Any,
    model: Any,
) -> tuple[type, dict[str, Any]]:
    """Resolve a configured optimizer for Trainer.optimizer_cls_and_kwargs."""
    name, configured_kwargs = _parse_optimizer_config(config)

    import torch

    kwargs: dict[str, Any] = {
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
    }

    if name in {
        "adam",
        "adam_bnb_8bit",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_bnb_8bit",
    }:
        kwargs.update(
            {
                "betas": (training_args.adam_beta1, training_args.adam_beta2),
                "eps": training_args.adam_epsilon,
            }
        )

    if name == "adam":
        optimizer_cls = torch.optim.Adam
    elif name == "adam_bnb_8bit":
        try:
            from bitsandbytes.optim import Adam
        except ImportError as exc:
            raise ImportError("optimizer adam_bnb_8bit requires bitsandbytes") from exc
        optimizer_cls = Adam
        kwargs["optim_bits"] = 8
        kwargs["is_paged"] = False
    elif name in {"adamw_torch", "adamw_torch_fused"}:
        optimizer_cls = torch.optim.AdamW
        if name == "adamw_torch_fused":
            kwargs["fused"] = True
    elif name == "adamw_bnb_8bit":
        try:
            from bitsandbytes.optim import AdamW
        except ImportError as exc:
            raise ImportError(
                "optimizer adamw_bnb_8bit requires bitsandbytes"
            ) from exc
        optimizer_cls = AdamW
        kwargs["optim_bits"] = 8
        kwargs["is_paged"] = False
    elif name in {"sgd", "sgd_momentum"}:
        optimizer_cls = torch.optim.SGD
        if name == "sgd_momentum":
            kwargs["momentum"] = 0.9
    else:
        optimizer_cls = getattr(torch.optim, "Muon", None)
        if optimizer_cls is None:
            raise RuntimeError("optimizer muon requires torch.optim.Muon (PyTorch 2.10+)")
        dimensions = _validate_muon_parameters(model)
        print(
            f"[train] Muon trainable parameter dimensions: {dict(sorted(dimensions.items()))}",
            flush=True,
        )

    kwargs.update(configured_kwargs)
    return optimizer_cls, kwargs
