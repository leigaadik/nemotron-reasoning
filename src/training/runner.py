"""Unified Transformers Trainer for pre-tokenized Qwen LoRA experiments."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import format_config, resolve_path
from src.data.jsonl import read_jsonl
from src.training.collator import ExplicitLabelsCollator
from src.training.stratified_sampler import build_stratified_index_order


def _install_runtime_compatibility() -> None:
    # The platform's optional FBGEMM Blackwell extension aborts while xFormers
    # probes it. This project does not use FBGEMM, so expose it as unavailable.
    sys.modules.setdefault("fbgemm_gpu", None)


def _patch_transformers_loader() -> None:
    import transformers

    original = transformers.AutoModelForCausalLM.from_pretrained.__func__

    @classmethod
    def patched(cls, *args, **kwargs):
        for key in (
            "unsloth_force_compile",
            "load_in_fp8",
            "unsloth_tiled_mlp",
            "fast_inference",
        ):
            kwargs.pop(key, None)
        return original(cls, *args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = patched


def _dtype(name: str):
    import torch

    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _load_training_dataset(path: Path):
    from datasets import load_dataset

    labels: list[str] = []
    for index, row in enumerate(read_jsonl(path)):
        input_ids = row.get("input_ids") or []
        token_labels = row.get("labels") or []
        if not input_ids or len(input_ids) != len(token_labels):
            raise ValueError(f"Invalid tokenized row {index} in {path}")
        if all(int(label) == -100 for label in token_labels):
            raise ValueError(f"Row {index} has no loss-bearing labels")
        labels.append(str(row.get("category", "unknown")))
    if not labels:
        raise ValueError(f"Training dataset is empty: {path}")
    dataset = load_dataset("json", data_files={"train": str(path)}, split="train")
    return dataset, labels


def train(config: dict[str, Any], config_path: Path) -> Path:
    _install_runtime_compatibility()
    import unsloth  # noqa: F401

    _patch_transformers_loader()
    from torch.utils.data import Sampler
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    experiment = config["experiment"]
    model_config = config["model"]
    lora_config = config["lora"]
    paths = config["paths"]
    trainer_config = dict(config["training"])
    optimizer_name = str(trainer_config.pop("optimizer", "adamw")).lower()
    muon_config = dict(trainer_config.pop("muon", {}))
    if optimizer_name not in {"adamw", "muon"}:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. Expected 'adamw' or 'muon'."
        )
    if optimizer_name == "muon":
        # These fields belong to AdamW and must not affect Muon experiments.
        for key in ("adam_beta1", "adam_beta2", "adam_epsilon"):
            trainer_config.pop(key, None)
    dataset_path = resolve_path(paths["train_jsonl"])
    dataset, categories = _load_training_dataset(dataset_path)

    print(f"[train] config: {config_path}", flush=True)
    print(format_config(config), flush=True)
    print(f"[train] dataset: {dataset_path}", flush=True)
    print(f"[train] rows: {len(dataset)}", flush=True)
    print(f"[train] categories: {dict(sorted(Counter(categories).items()))}", flush=True)

    model_path = str(resolve_path(model_config["model_path"]))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=int(model_config["max_seq_length"]),
        load_in_4bit=bool(model_config.get("load_in_4bit", False)),
        load_in_8bit=bool(model_config.get("load_in_8bit", False)),
        full_finetuning=False,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        unsloth_force_compile=bool(model_config.get("unsloth_force_compile", False)),
        attn_implementation=model_config.get("attn_implementation", "eager"),
        dtype=_dtype(model_config.get("dtype", "bfloat16")),
    )
    if tokenizer.eos_token not in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.pad_token = "<|endoftext|>"

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(experiment["seed"]),
    )
    model.print_trainable_parameters()

    trainer_config["output_dir"] = str(resolve_path(trainer_config["output_dir"]))
    trainer_config["seed"] = int(experiment["seed"])
    training_args = TrainingArguments(**trainer_config)
    effective_batch = (
        int(training_args.per_device_train_batch_size)
        * int(training_args.gradient_accumulation_steps)
    )
    order = build_stratified_index_order(
        categories,
        effective_batch,
        int(experiment["seed"]),
    )

    class OrderedSampler(Sampler):
        def __iter__(self):
            return iter(order)

        def __len__(self):
            return len(order)

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return OrderedSampler()

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            if optimizer_name == "adamw":
                return super().create_optimizer()

            import torch

            trainable = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            invalid = [
                (name, tuple(parameter.shape))
                for name, parameter in trainable
                if parameter.ndim != 2
            ]
            if invalid:
                preview = invalid[:8]
                raise ValueError(
                    "Muon requires every trainable parameter to be 2D; "
                    f"invalid parameters: {preview}"
                )

            muon_kwargs = dict(muon_config)
            muon_kwargs.pop("lr", None)
            muon_kwargs.pop("weight_decay", None)
            self.optimizer = torch.optim.Muon(
                [parameter for _, parameter in trainable],
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                **muon_kwargs,
            )
            print(
                f"[train] optimizer: Muon ({len(trainable)} trainable tensors)",
                flush=True,
            )
            return self.optimizer

    trainer = OrderedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ExplicitLabelsCollator(pad_token_id=int(tokenizer.pad_token_id)),
    )
    print(f"[train] effective batch size: {effective_batch}", flush=True)
    print("[train] starting", flush=True)
    started = time.time()
    trainer.train()
    print(f"[train] completed in {(time.time() - started) / 60:.1f} min", flush=True)

    adapter_dir = resolve_path(paths["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[train] adapter: {adapter_dir}", flush=True)
    return adapter_dir
