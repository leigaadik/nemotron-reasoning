# LoRA 微调实验说明

本文档说明 `exp/lora-finetuning` 分支中 LoRA 训练与评估的当前实现。当前训练主线使用 Qwen3-30B-A3B。

## 实验目标

本分支用于训练 Qwen3-30B-A3B 的 LoRA adapter，并在固定验证集上评估训练效果。训练数据来自 Nemotron Reasoning Challenge 的原始题目和增强 CoT 数据。

## 数据

当前使用的数据文件：

- `data/train.csv`：原始训练集，包含 `id`、`prompt`、`answer` 字段。
- `data/train_split_with_cot.csv`：LoRA SFT 训练数据，包含 `id`、`prompt`、`answer`、`type`、`generated_cot` 字段。
- `configs/eval/validation_ids_seed42_size950.csv`：固定 950 题验证集。

SFT 样本由 `src/data/sft_dataset.py` 构造：

- user：原始 `prompt` 加 boxed answer 格式提示。
- assistant：清洗后的 `generated_cot`、`</think>`、最终 `\boxed{answer}`。

## 当前项目结构

```text
configs/
  eval/
    validation_ids_seed42_size950.csv
  training/
    lora_unsloth_qwen3_30b_a3b.yaml
scripts/
  train_lora_unsloth.py
  evaluate_adapter.py
  evaluate_baseline.py
src/
  data/
    sft_dataset.py
  evaluation/
    categories.py
    scoring.py
  prompting/
    dataset.py
  providers/
    base.py
    local_vllm.py
  training/
    config.py
    stratified_sampler.py
```

各模块职责：

- `scripts/train_lora_unsloth.py`：LoRA 训练入口，读取 YAML 配置，构造 SFT 数据，加载 Unsloth 模型，创建 LoRA adapter，并保存训练产物。
- `scripts/evaluate_adapter.py`：使用 vLLM 加载 base model 与 LoRA adapter，生成验证集回答。
- `scripts/evaluate_baseline.py`：读取生成结果并打分。
- `src/data/sft_dataset.py`：将 `train_split_with_cot.csv` 转换为 TRL SFTTrainer 需要的 chat records。
- `src/training/config.py`：加载训练 YAML，并解析仓库内相对路径。
- `src/training/stratified_sampler.py`：按题目 `type` 构造近似分层训练顺序。
- `src/evaluation/`：答案抽取、类别统计和评分逻辑。
- `src/prompting/dataset.py`：读取固定验证集题目。
- `src/providers/local_vllm.py`：本地 vLLM 推理封装，支持 LoRA adapter。

## Qwen3-30B-A3B 训练配置

主配置文件：

```text
configs/training/lora_unsloth_qwen3_30b_a3b.yaml
```

核心配置：

- base model：`Qwen/Qwen3-30B-A3B`
- 本地模型路径：`./models/Qwen3-30B-A3B`
- 训练数据：`data/train_split_with_cot.csv`
- 输出目录：`outputs/lora_finetuning/qwen3_30b_a3b`
- adapter 目录：`outputs/lora_finetuning/qwen3_30b_a3b/adapter`
- 模型加载长度：`max_seq_length=8192`
- SFT 截断长度：`max_length=4096`
- 精度：`bfloat16`
- 量化加载：不使用 4bit / 8bit
- LoRA rank：`r=32`
- LoRA alpha：`lora_alpha=32`
- LoRA dropout：`0.0`
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`
- epoch：`1`
- batch size：`per_device_train_batch_size=1`
- 梯度累积：`gradient_accumulation_steps=8`
- 学习率：`2.0e-4`
- packing：`false`
- Liger kernel：`use_liger_kernel=true`

## 启动训练

以下命令在项目根目录执行：

```bash
conda activate nemotron

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ACCELERATE_BYPASS_DEVICE_MAP=true \
PYTHONUNBUFFERED=1 \
python scripts/train_lora_unsloth.py \
  --config configs/training/lora_unsloth_qwen3_30b_a3b.yaml
```

## Python 依赖

训练环境使用 Python 3.10、CUDA 12.1、PyTorch 2.5.1。核心依赖记录在 `requirements.txt`：

```text
accelerate==1.14.0
bitsandbytes==0.49.2
datasets==3.6.0
liger-kernel==0.8.1
pandas==2.3.3
peft==0.19.1
PyYAML==6.0.3
torch==2.5.1+cu121
transformers==4.57.3
triton==3.1.0
trl==0.24.0
unsloth==2026.7.4
unsloth-zoo==2026.7.4
```

## Adapter 评估流程

训练完成后，用 vLLM 加载 Qwen3-30B-A3B base model，并挂载训练得到的 LoRA adapter，在固定 950 题验证集上生成回答，再运行评分脚本打分。

```text
outputs/lora_finetuning/qwen3_30b_a3b/adapter/
        |
        |  scripts/evaluate_adapter.py
        v
results/lora_finetuning/qwen3-30b-a3b-lora/
    qwen3-30b-a3b-lora_raw_outputs.jsonl
    qwen3-30b-a3b-lora_run.yaml
        |
        |  scripts/evaluate_baseline.py
        v
    qwen3-30b-a3b-lora_validation.csv
    qwen3-30b-a3b-lora_results.csv
    qwen3-30b-a3b-lora_mistakes/
```

完整评估命令：

以下命令在项目根目录执行：

```bash
python scripts/evaluate_adapter.py \
  --model-name qwen3-30b-a3b-lora \
  --base-model ./models/Qwen3-30B-A3B \
  --adapter outputs/lora_finetuning/qwen3_30b_a3b/adapter \
  --out results/lora_finetuning/qwen3-30b-a3b-lora

python scripts/evaluate_baseline.py \
  --run-dir results/lora_finetuning/qwen3-30b-a3b-lora
```

推理参数：

```text
temperature: 0.0
top_p: 1.0
max_tokens: 32768
max_model_len: 32768
enable_thinking: true
prompt_suffix: Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

## 训练数据 CoT token 长度分布

用 Qwen3-30B-A3B tokenizer 对 9500 条完整 SFT 序列（user prompt + CoT + `</think>` + `\boxed{}`，chat template 后）统计 token 长度：

| 类别 | 条数 | 均值 | 中位数 | p90 | p95 | max | %>4096 | %>7680 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **总体** | 9500 | 3442 | 3028 | 6838 | 7184 | 7958 | 33.7% | 0.1% |
| bit_manipulation | 1602 | 6943 | 6937 | 7451 | 7524 | 7958 | 100.0% | 0.8% |
| equation_numeric_guess | 136 | 6134 | 6110 | 6487 | 6639 | 6847 | 100.0% | 0.0% |
| equation_numeric_deduce | 596 | 5880 | 5818 | 6224 | 6322 | 6992 | 100.0% | 0.0% |
| gravity | 1597 | 3699 | 3618 | 4864 | 5182 | 6294 | 32.4% | 0.0% |
| cipher | 1576 | 3241 | 3110 | 4607 | 5130 | 7120 | 20.4% | 0.0% |
| unit_conversion | 1594 | 2529 | 2466 | 3488 | 3769 | 4736 | 1.7% | 0.0% |
| numeral | 1576 | 1048 | 1044 | 1088 | 1101 | 1138 | 0.0% | 0.0% |
| cryptarithm_deduce | 659 | 660 | 652 | 763 | 768 | 779 | 0.0% | 0.0% |
| cryptarithm_guess | 164 | 645 | 650 | 760 | 765 | 780 | 0.0% | 0.0% |

总体分布：

![SFT token length overall](figures/token_length_overall.png)

各类别分布（红线=4096，绿线=7680）：

![SFT token length by category](figures/token_length_by_category.png)

要点：bit_manipulation 的 CoT 最长（中位数 6937、max 7958），是决定 max_length 的卡点——4096 截掉其 100% 样本、7680 仅剩 0.8%；cryptarithm_deduce/guess 反而最短（约 650 tokens），任何 max_length 都不截断，其低分属能力/数据问题而非截断。

## 微调前后对比

微调前为 Qwen3-30B-A3B **零样本**，评估 `max_tokens=32768`；微调后为 **ml=7680** adapter，评估采用竞赛口径 `max_tokens=7680 / max_model_len=8192`。

| category | 微调前（零样本，32768） | 微调后（ml=7680） |
|---|---:|---:|
| numeral | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% |
| gravity | 99.4% | 100.0% |
| cipher | 42.7% | 99.4% |
| bit_manipulation | 26.2% | 81.9% |
| equation_numeric_deduce | 53.3% | 90.0% |
| equation_numeric_guess | 7.1% | 7.1% |
| cryptarithm_deduce | 3.0% | 6.1% |
| cryptarithm_guess | 0.0% | 0.0% |
| **TOTAL** | **65.3% (620/950)** | **86.6% (823/950)** |

## 实验记录：max_length 消融

> 训练设备：**NVIDIA H200（单卡）**。

核心参数配置如下：

| 参数 | 取值 |
|---|---|
| base model | Qwen/Qwen3-30B-A3B |
| LoRA rank / alpha / dropout | 32 / 32 / 0.0 |
| learning_rate | 2e-4 |
| epochs | 1 |
| batch_size × grad_accum | 1 × 8 |
| 训练数据 | train_split_with_cot.csv（9500 条） |
| **训练 max_length** | **4096 / 7680 / 8192** |
| 评估 max_model_len | 8192 |
| 评估 max_tokens | 7680 |
| temperature / top_p | 0.0 / 1.0 |
| enable_thinking | true |
| 验证集 | 训练集抽样 950 题 |

评估统一采用竞赛口径（base + adapter 经 vLLM 加载生成，与官方评测一致）。三组实验仅训练 `max_length` 取值不同，其余训练与推理配置完全一致。

各类别准确率：

| category | ml=4096 | ml=7680 | ml=8192 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 100.0% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 99.4% | 99.4% | 98.7% |
| bit_manipulation | 0.0% | 81.9% | 80.0% |
| equation_numeric_deduce | 13.3% | 90.0% | 88.3% |
| equation_numeric_guess | 0.0% | 7.1% | 7.1% |
| cryptarithm_deduce | 4.5% | 6.1% | 4.5% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **67.8% (644/950)** | **86.6% (823/950)** | **86.0% (817/950)** |

## 实验记录：batch size 消融（固定 max_length=7680）

> 训练设备：**NVIDIA H200（单卡）**。

固定 max_length=7680、lr=2e-4、max_grad_norm=1e9，其余同基础配置；唯一变量为**有效 batch**（per_device=1 × gradient_accumulation_steps）。评估统一竞赛口径（max_model_len=8192 / max_tokens=7680）。

| category | eff_batch=16 | eff_batch=8 | eff_batch=4 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 100.0% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 100.0% | 98.7% | 99.4% |
| bit_manipulation | 74.4% | 80.0% | 83.8% |
| equation_numeric_deduce | 85.0% | 88.3% | 88.3% |
| equation_numeric_guess | 0.0% | 7.1% | 7.1% |
| cryptarithm_deduce | 6.1% | 6.1% | 7.6% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **85.1% (808/950)** | **86.1% (818/950)** | **86.9% (826/950)** |

## 实验记录：learning rate 消融（固定 max_length=7680、eff_batch=8）

> 训练设备：**NVIDIA B200（单卡）**。

固定 max_length=7680、eff_batch=8、max_grad_norm=1e9，其余同基础配置；唯一变量为 **learning_rate**。评估统一竞赛口径（max_model_len=8192 / max_tokens=7680）。

| category | lr=1e-4 | lr=2e-4 | lr=5e-4 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 99.4% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 98.7% | 99.4% | 100.0% |
| bit_manipulation | 77.5% | 80.6% | 80.6% |
| equation_numeric_deduce | 86.7% | 86.7% | 91.7% |
| equation_numeric_guess | 7.1% | 7.1% | 7.1% |
| cryptarithm_deduce | 6.1% | 6.1% | 9.1% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **85.5% (812/950)** | **86.2% (819/950)** | **86.8% (825/950)** |

## 实验记录：r / alpha 网格搜索（固定 max_length=7680、eff_batch=4、lr=2e-4）

> 训练设备：**NVIDIA H200（单卡）**。

固定 max_length=7680、eff_batch=4、lr=2e-4，其余同基础配置；网格搜索 LoRA rank `r` 与 `alpha` ∈ {16, 32, 64}（共 9 组，scaling = alpha/r）。评估口径 max_model_len=8192 / max_tokens=7680；评估的 `max_lora_rank` 设为 64。

TOTAL 准确率（行 = r，列 = alpha）：

| r ＼ alpha | 16 | 32 | 64 |
|---|---:|---:|---:|
| **16** | 86.5% | **87.1%** | 86.9% |
| **32** | 86.3% | 86.9% | **87.1%** |
| **64** | 86.3% | 86.1% | **86.6%** |


### r=16 各类别明细

| category | alpha=16 | alpha=32 | alpha=64 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 100.0% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 100.0% | 100.0% | 100.0% |
| bit_manipulation | 81.9% | 83.1% | 83.8% |
| equation_numeric_deduce | 88.3% | 88.3% | 86.7% |
| equation_numeric_guess | 7.1% | 14.3% | 7.1% |
| cryptarithm_deduce | 4.5% | 7.6% | 7.6% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **86.5% (822/950)** | **87.1% (827/950)** | **86.9% (826/950)** |

### r=32 各类别明细

| category | alpha=16 | alpha=32 | alpha=64 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 100.0% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 99.4% | 99.4% | 100.0% |
| bit_manipulation | 80.6% | 83.8% | 83.8% |
| equation_numeric_deduce | 88.3% | 88.3% | 85.0% |
| equation_numeric_guess | 7.1% | 7.1% | 14.3% |
| cryptarithm_deduce | 6.1% | 7.6% | 9.1% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **86.3% (820/950)** | **86.9% (826/950)** | **87.1% (827/950)** |

### r=64 各类别明细

| category | alpha=16 | alpha=32 | alpha=64 |
|---|---:|---:|---:|
| numeral | 100.0% | 100.0% | 100.0% |
| gravity | 100.0% | 100.0% | 100.0% |
| unit_conversion | 100.0% | 100.0% | 100.0% |
| cipher | 100.0% | 98.7% | 100.0% |
| bit_manipulation | 80.6% | 80.6% | 81.9% |
| equation_numeric_deduce | 86.7% | 86.7% | 86.7% |
| equation_numeric_guess | 7.1% | 7.1% | 7.1% |
| cryptarithm_deduce | 6.1% | 6.1% | 7.6% |
| cryptarithm_guess | 0.0% | 0.0% | 0.0% |
| **TOTAL** | **86.3% (820/950)** | **86.1% (818/950)** | **86.6% (823/950)** |

## 兼容性问题与处理

### Unsloth 参数透传

当前 Unsloth 版本可能向 Hugging Face `AutoModelForCausalLM.from_pretrained` 透传模型构造函数不接受的参数：

- `unsloth_force_compile`
- `load_in_fp8`
- `unsloth_tiled_mlp`
- `fast_inference`

训练脚本会在模型加载前过滤这些参数。

### Qwen tokenizer 特殊 token

Unsloth / TRL 组合中可能出现占位 token：

- `<EOS_TOKEN>`
- `<PAD_TOKEN>`

训练脚本会将它们映射到 Qwen tokenizer 中可用的真实 token，避免 TRL 初始化和数据处理阶段的 token 校验失败。

### TRL 0.24 logits entropy 路径

TRL 0.24 的 `SFTTrainer.compute_loss` 会读取 `outputs.logits` 计算 entropy。Qwen3 配置启用 `use_liger_kernel=true`，用于避开该路径下的兼容性问题。

### torchao 与 torch 版本不兼容

如果导入 `trl.SFTTrainer` 或 `transformers.AutoProcessor` 时出现：

```text
ImportError: cannot import name 'AutoProcessor' from 'transformers'
RuntimeError: Failed to import trl.trainer.sft_trainer
AttributeError: module 'torch' has no attribute 'int1'
```

通常是环境中额外安装的 `torchao` 版本与当前 `torch==2.5.1+cu121` 不兼容。当前训练链路不依赖 `torchao`，可以卸载：

```bash
pip uninstall -y torchao
```

### 本地模型加载

Qwen3-30B-A3B 配置使用本地模型路径：

```text
./models/Qwen3-30B-A3B
```

训练时设置：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

用于避免 Hugging Face Hub 联网探测本地模型和 adapter 路径。

### Accelerate device map 检查

Unsloth 加载模型时可能使用 `device_map=auto`。单进程训练时设置：

```bash
ACCELERATE_BYPASS_DEVICE_MAP=true
```

用于绕过 Accelerate 对 device map 的训练限制检查。
