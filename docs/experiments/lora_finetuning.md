# LoRA 微调实验说明

本文档记录 `exp/lora-finetuning` 分支的开发目标、实验流程和阶段性记录。本分支专注于复现并改进 NVIDIA Nemotron Reasoning Challenge 的 LoRA 微调方案。

## 分支目标

本分支用于开发 LoRA 微调相关代码和实验资产，目标是在不提交完整模型权重的前提下，训练并产出符合竞赛要求的 LoRA adapter，提升模型在逻辑推理 benchmark 上的准确率。

核心问题包括：

- 选择合适的基座模型和训练框架。
- 使用官方训练数据和清洗后的高质量数据构造训练样本。
- 设计稳定的验证集切分和评估流程。
- 训练 LoRA adapter 并和零样本 prompting baseline 对比。
- 整理最终可提交的 `submission.zip`。

## 数据说明

当前项目中的主要数据：

- `data/train.csv`：竞赛官方原始训练集，包含 `id`、`prompt`、`answer` 字段。
- `data/train_split_with_cot.csv`：基于原始数据清洗或增强得到的高质量训练数据，可用于 LoRA 微调实验。

建议在训练前固定验证集划分，避免后续调参时评估结果不可比较。

## 初始开发任务

1. 梳理并迁移可复现的 Unsloth 训练 notebook。
2. 将 notebook 中的关键训练逻辑整理为可维护脚本。
3. 固定训练集和验证集划分。
4. 实现 LoRA 训练配置，包括模型路径、LoRA rank、学习率、batch size、max sequence length 等。
5. 接入验证脚本，评估 adapter 在保留验证集上的准确率。
6. 保存实验结果，包括训练配置、日志、验证分数和 adapter 路径。

## Notebook 参考方案

当前参考 notebook 的核心 LoRA 方案如下：

- 基座模型：`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`。
- 训练框架：Unsloth `FastLanguageModel` + TRL `SFTTrainer` / `SFTConfig`。
- 精度：`torch.bfloat16`，不使用 4bit / 8bit 量化加载。
- 最大上下文：模型加载 `max_seq_length=8192`，SFT 训练 `max_length=4096`。
- LoRA 配置：`r=32`，`lora_alpha=32`，`lora_dropout=0.0`，`bias="none"`。
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`in_proj`、`out_proj`、`up_proj`、`down_proj`、`lm_head`。
- 训练数据：读取包含 `prompt`、`answer`、`generated_cot`、`type` 的高质量 CoT 数据。
- 样本格式：user 为原始 `prompt` 加 boxed answer 提示，assistant 为清洗后的 CoT 加最终 `\boxed{answer}`。
- 训练参数：`num_train_epochs=1`，`per_device_train_batch_size=1`，`gradient_accumulation_steps=8`，`learning_rate=2e-4`，`bf16=True`，`packing=False`。
- 采样策略：按题目 `type` 构造近似分层顺序，尽量让每个有效 batch 的题型分布更均衡。
- 产物：保存 LoRA adapter，并将 `adapter_config.json` 和 `adapter_model.safetensors` 打包为 `submission.zip`。

这个方案应先脚本化复现，再逐步替换 notebook 中的 Kaggle 固定路径和手写参数。

## 配置管理

训练超参数统一写入 YAML：

```text
configs/training/lora_unsloth_nemotron_30b_a3b.yaml
```

训练入口读取该 YAML，并在启动时打印解析后的配置路径和完整参数快照：

```bash
python scripts/train_lora_unsloth.py \
  --config configs/training/lora_unsloth_nemotron_30b_a3b.yaml
```

训练日志开头会明确记录：

- 实际读取的 YAML 绝对路径。
- 基座模型和本地模型路径。
- 数据路径。
- LoRA 参数。
- SFT 训练参数。
- 输出目录、adapter 目录和 submission 路径。

## 项目框架设计

LoRA 分支建议采用“公共评估能力 + 独立训练能力”的结构。零样本 baseline 和 LoRA 微调都需要读取同一批验证题、使用同一套答案抽取和 scoring，因此这些部分应该兼容共享；训练、adapter 打包和训练配置则单独放在 LoRA 分支内。

推荐结构：

```text
configs/
  eval/
    validation_ids_seed42_size950.csv
  training/
    lora_unsloth_nemotron_30b_a3b.yaml
scripts/
  train_lora_unsloth.py
  package_lora_submission.py
  evaluate_adapter.py
src/
  data/
    sft_dataset.py
  evaluation/
    scoring.py
    categories.py
  training/
    lora_config.py
    stratified_sampler.py
  prompting/
    dataset.py
notebooks/
  training/
    unsloth_lora_training.ipynb
  evaluation/
    adapter_validation.ipynb
    nvidia_nemotron_metric.ipynb
```

其中：

- `src/evaluation/` 应和 `exp/prompting-baselines` 保持一致，用于保证 baseline 和 LoRA 分数可比。
- `src/prompting/dataset.py` 可以继续负责固定验证集读取。
- `src/data/sft_dataset.py` 负责把 `train_split_with_cot.csv` 转成 SFT messages。
- `src/training/stratified_sampler.py` 从 notebook 中抽出分层 batch 顺序逻辑。
- `scripts/train_lora_unsloth.py` 负责读取 YAML、构造数据、加载 Unsloth 模型、创建 LoRA、启动 TRL SFT 训练并保存 adapter。
- `scripts/package_lora_submission.py` 只负责生成竞赛需要的 `submission.zip`。
- `scripts/evaluate_adapter.py` 负责加载 base model + adapter，在固定验证集上生成回答并复用 `src/evaluation/scoring.py` 打分。

## 与另一个实验分支的兼容关系

不建议在 `exp/lora-finetuning` 上直接修改 `exp/prompting-baselines` 的项目结构。更合理的方式是：

1. 保持 `exp/prompting-baselines` 的零样本推理代码稳定。
2. 在 LoRA 分支中复用或迁移公共模块，比如 `src/evaluation/`、`src/prompting/dataset.py`、`configs/eval/validation_ids_seed42_size950.csv`。
3. 如果公共模块确实需要调整，先在当前分支改好并验证，再合并到 `master`，最后让两个实验分支都从 `master` 同步。

这样做的好处是两个实验分支可以共享评估口径，但不会互相打断开发节奏。LoRA 分支新增训练相关目录和脚本即可，不需要重构 prompting 分支已有代码。


## Qwen3-4B 本地 LoRA 调试配置

为降低本地依赖和显存调试成本，本分支新增 Qwen3-4B LoRA 配置：

```text
configs/training/lora_unsloth_qwen3_4b.yaml
```

该配置用于在本地 `models/Qwen3-4B` 权重上验证 Unsloth + TRL 训练链路，主要差异如下：

- 基座模型：`Qwen/Qwen3-4B`，本地路径 `models/Qwen3-4B`。
- 最大上下文：模型加载 `max_seq_length=4096`，SFT 训练 `max_length=2048`。
- LoRA 配置：`r=16`，`lora_alpha=16`，`lora_dropout=0.0`。
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- 数据预处理：`dataset_num_proc=1`，避免 tokenizer 兼容补丁在多进程 `datasets.map` 中触发 pickle 问题。
- 训练：`use_liger_kernel=true`，避开 TRL 0.24 对 `outputs.logits` 的额外 entropy 统计路径。

启动命令建议显式指定本地模型离线加载和单进程 device map 绕过：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ACCELERATE_BYPASS_DEVICE_MAP=true \
python scripts/train_lora_unsloth.py \
  --config configs/training/lora_unsloth_qwen3_4b.yaml
```

其中 `CUDA_VISIBLE_DEVICES` 应根据 `nvidia-smi` 选择空闲 GPU。

## Python 依赖组合

当前 Python 3.10 环境采用 CUDA 12.1 / PyTorch 2.5.1 路线，核心依赖记录在：

```text
requirements.txt
```

依赖文件由 `pipreqs --use-local` 基于源码 import 生成后，补充运行期必需包得到。核心版本包括：

- `torch==2.5.1+cu121`
- `transformers==4.57.3`
- `trl==0.24.0`
- `peft==0.19.1`
- `datasets==3.6.0`
- `accelerate==1.14.0`
- `liger-kernel==0.8.1`
- `unsloth==2026.7.4`
- `unsloth-zoo==2026.7.4`

不建议把完整 `pip freeze` 直接作为项目依赖文件。当前训练路径需要的是精简、可复现的训练依赖；推理侧 `vllm` 依赖与训练侧 CUDA / PyTorch 版本容易互相牵制，建议单独环境维护。

## 兼容性问题与处理

### Unsloth 参数透传

当前 Unsloth 版本会向 Hugging Face `AutoModelForCausalLM.from_pretrained` 透传部分模型构造函数不接受的参数，例如：

- `unsloth_force_compile`
- `load_in_fp8`
- `unsloth_tiled_mlp`
- `fast_inference`

训练入口在加载模型前临时过滤这些参数，避免 Qwen3 模型初始化时报 `unexpected keyword argument`。

### Qwen tokenizer 特殊 token

Unsloth / TRL 组合中可能出现占位 token：

- `<EOS_TOKEN>`
- `<PAD_TOKEN>`

Qwen3 tokenizer 的真实终止符和 padding 相关 token 分别使用：

- `eos_token`: `<|im_end|>`
- `pad_token`: `<|endoftext|>`

训练脚本会在模型加载和 LoRA 注入后修正 tokenizer，并在 TRL 初始化前把占位 token 映射到真实 token id，避免 TRL 的 token 校验失败。

### TRL 0.24 logits entropy 路径

TRL 0.24 的 `SFTTrainer.compute_loss` 会额外读取 `outputs.logits` 计算 entropy。Unsloth Qwen3 patch 下该字段可能不是常规 tensor。Qwen 配置启用 `use_liger_kernel=true`，让 TRL 跳过这条额外统计路径。

### 本地模型加载

Qwen 调试配置使用本地模型路径 `models/Qwen3-4B`。运行时建议设置：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

这样可以避免 Hugging Face Hub 对 `adapter_config.json` 等文件进行联网探测。

### Accelerate device map 检查

Unsloth 加载模型时可能使用 `device_map=auto`。在单进程训练时，Accelerate 仍可能触发 device map 检查。运行时设置：

```bash
ACCELERATE_BYPASS_DEVICE_MAP=true
```

用于明确允许该单进程训练路径继续执行。

## 实验记录

| 日期 | 模型 | 框架 | 数据 | 主要配置 | 验证分数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | Unsloth / TRL | TBD | TBD | TBD | 初始实验 |

## 与零样本 baseline 的关系

`exp/prompting-baselines` 分支用于测试不同大模型在无微调、仅提示词引导下的推理能力。本分支的 LoRA 微调结果应和该 baseline 进行对比，判断微调是否带来稳定收益。

评估时需要尽量保持：

- 相同或可解释的验证集。
- 相同的答案抽取规则。
- 相同的 scoring 方法。
- 清晰记录是否使用 CoT、是否限制输出长度、是否使用官方提示格式。
