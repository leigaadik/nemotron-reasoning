#!/usr/bin/env python3
"""Train a LoRA adapter with Unsloth + TRL SFTTrainer."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sft_dataset import build_sft_records, to_hf_dataset  # noqa: E402
from src.training.config import format_config, load_training_config  # noqa: E402
from src.training.config import resolve_repo_path  # noqa: E402
from src.training.stratified_sampler import build_stratified_index_order  # noqa: E402


DEFAULT_CONFIG = "configs/training/lora_unsloth_nemotron_30b_a3b.yaml"

# ==================== 动态修复 Unsloth 参数转发 Bug ====================
import transformers

# 备份原始的 from_pretrained 类方法底层函数
original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained.__func__

@classmethod
def patched_from_pretrained(cls, *args, **kwargs):
    # 拦截并安全剔除无法被 Hugging Face 模型构造函数识别的 Unsloth 专属参数
    unwanted_keys = [
        "unsloth_force_compile",
        "load_in_fp8",
        "unsloth_tiled_mlp",
        "fast_inference"
    ]
    for key in unwanted_keys:
        kwargs.pop(key, None)
    return original_from_pretrained(cls, *args, **kwargs)

# 动态替换
transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Training YAML config path. Default: {DEFAULT_CONFIG}",
    )
    return parser.parse_args()


def _dtype_from_config(dtype_name: str):
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _print_config(config: dict[str, Any], config_path: Path) -> None:
    print(f"[train_lora_unsloth] config_path: {config_path}", flush=True)
    print("[train_lora_unsloth] config:", flush=True)
    print(format_config(config), flush=True)


def _build_data(config: dict[str, Any]):
    experiment_cfg = config["experiment"]
    paths_cfg = config["paths"]
    prompt_cfg = config["prompt"]
    data_cfg = config["data"]
    train_csv = resolve_repo_path(paths_cfg["train_csv"])

    result = build_sft_records(
        train_csv,
        prompt_column=data_cfg["prompt_column"],
        answer_column=data_cfg["answer_column"],
        cot_column=data_cfg["cot_column"],
        type_column=data_cfg["type_column"],
        prompt_suffix=prompt_cfg["suffix"],
        assistant_think_end_token=prompt_cfg["assistant_think_end_token"],
        min_cot_chars=int(data_cfg["min_cot_chars"]),
        shuffle=bool(data_cfg["shuffle"]),
        seed=int(experiment_cfg["seed"]),
    )
    print(f"[train_lora_unsloth] train_csv: {train_csv}", flush=True)
    print(f"[train_lora_unsloth] sft_records: {len(result.records)}", flush=True)
    print(
        f"[train_lora_unsloth] type_counts: {result.type_counts}",
        flush=True,
    )
    return result


def _formatting_func(tokenizer, enable_thinking: bool):
    def formatting_prompts_func(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts

    return formatting_prompts_func


def _build_trainer(config: dict[str, Any], data_result):
    from torch.utils.data import DataLoader, Sampler
    from trl import SFTConfig, SFTTrainer

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    prompt_cfg = config["prompt"]
    training_cfg = dict(config["training"])
    experiment_cfg = config["experiment"]
    paths_cfg = config["paths"]

    import torch
    from unsloth import FastLanguageModel

    model_name = model_cfg.get("model_path") or model_cfg["base_model_name"]
    print(f"[train_lora_unsloth] model_name: {model_name}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(model_cfg["max_seq_length"]),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
        load_in_8bit=bool(model_cfg["load_in_8bit"]),
        full_finetuning=bool(model_cfg["full_finetuning"]),
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
        unsloth_force_compile=bool(model_cfg["unsloth_force_compile"]),
        attn_implementation=model_cfg["attn_implementation"],
        dtype=_dtype_from_config(model_cfg["dtype"]),
    )
    if tokenizer.eos_token == "<EOS_TOKEN>" or tokenizer.eos_token not in tokenizer.get_vocab():
        for token in ["<|im_end|>", "<|endoftext|>"]:
            if token in tokenizer.get_vocab():
                tokenizer.eos_token = token
                break
    if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"[train_lora_unsloth] tokenizer eos_token={tokenizer.eos_token!r} "
        f"pad_token={tokenizer.pad_token!r}",
        flush=True,
    )
    print("[train_lora_unsloth] model loaded with Unsloth.", flush=True)

    print("[train_lora_unsloth] creating LoRA wrapper.", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=int(experiment_cfg["seed"]),
    )
    model.print_trainable_parameters()
    if tokenizer.eos_token == "<EOS_TOKEN>" or tokenizer.eos_token not in tokenizer.get_vocab():
        for token in ["<|im_end|>", "<|endoftext|>"]:
            if token in tokenizer.get_vocab():
                tokenizer.eos_token = token
                break
    if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.pad_token = tokenizer.eos_token
    original_convert_tokens_to_ids = tokenizer.convert_tokens_to_ids

    def convert_tokens_to_ids_with_unsloth_tokens(token):
        if token == "<EOS_TOKEN>":
            return original_convert_tokens_to_ids(tokenizer.eos_token)
        if token == "<PAD_TOKEN>":
            return original_convert_tokens_to_ids(tokenizer.pad_token)
        if isinstance(token, list):
            return [convert_tokens_to_ids_with_unsloth_tokens(item) for item in token]
        return original_convert_tokens_to_ids(token)

    tokenizer.convert_tokens_to_ids = convert_tokens_to_ids_with_unsloth_tokens
    training_cfg["output_dir"] = str(resolve_repo_path(training_cfg["output_dir"]))
    training_cfg["seed"] = int(experiment_cfg["seed"])
    training_cfg.pop("eos_token", None)
    training_cfg.pop("pad_token", None)
    training_args = SFTConfig(**training_cfg)
    class PrecomputedOrderSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order=None, **kwargs):
            trainer_args = kwargs.get("args")
            processing_class = kwargs.get("processing_class")
            if trainer_args is not None and processing_class is not None:
                eos_token = processing_class.eos_token
                if eos_token == "<EOS_TOKEN>" or eos_token not in processing_class.get_vocab():
                    eos_token = "<|im_end|>"
                    processing_class.eos_token = eos_token
                pad_token = processing_class.pad_token
                if pad_token == "<PAD_TOKEN>" or pad_token not in processing_class.get_vocab():
                    pad_token = "<|endoftext|>" if "<|endoftext|>" in processing_class.get_vocab() else eos_token
                    processing_class.pad_token = pad_token
                trainer_args.eos_token = eos_token
                trainer_args.pad_token = pad_token
                print(
                    "[train_lora_unsloth] trainer init eos_token={!r} pad_token={!r}".format(
                        trainer_args.eos_token, trainer_args.pad_token
                    ),
                    flush=True,
                )
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            if len(self.stratified_order) != len(self.train_dataset):
                raise ValueError("Stratified order length does not match train dataset")

            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )
            return DataLoader(self.train_dataset, **dataloader_kwargs)

    dataset = to_hf_dataset(data_result.records)
    effective_batch_size = max(
        1,
        int(training_args.per_device_train_batch_size)
        * int(training_args.gradient_accumulation_steps),
    )
    stratified_order = None
    if config["data"]["stratify_by_type"]:
        stratified_order = build_stratified_index_order(
            data_result.labels,
            effective_batch_size,
            int(experiment_cfg["seed"]),
        )

    print(
        f"[train_lora_unsloth] effective_batch_size: {effective_batch_size}",
        flush=True,
    )
    print(
        f"[train_lora_unsloth] stratified_order: "
        f"{'enabled' if stratified_order is not None else 'disabled'}",
        flush=True,
    )

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=_formatting_func(
            tokenizer,
            enable_thinking=bool(prompt_cfg["enable_thinking"]),
        ),
        stratified_order=stratified_order,
    )
    adapter_dir = resolve_repo_path(paths_cfg["adapter_dir"])
    return trainer, model, tokenizer, adapter_dir


def main() -> None:
    args = parse_args()
    config, config_path = load_training_config(args.config)
    _print_config(config, config_path)

    data_result = _build_data(config)
    trainer, model, tokenizer, adapter_dir = _build_trainer(config, data_result)

    print("[train_lora_unsloth] starting SFT training.", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train_lora_unsloth] training done in {elapsed / 60:.1f} min", flush=True)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[train_lora_unsloth] adapter saved to: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
