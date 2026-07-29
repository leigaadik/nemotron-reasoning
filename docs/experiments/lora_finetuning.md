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

## 实验记录

| 日期 | 模型 | 框架 | 数据 | 主要配置 | 验证分数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-24 | Qwen3-30B-A3B | Unsloth / TRL | train_split_with_cot.csv | r=32, max_length=4096, lr=2e-4 | **69.7%** (662/950) | cipher 大幅提升；bit_manipulation / equation_numeric_deduce 明显退化，详见下方明细 |

### Qwen3-30B-A3B LoRA 分类明细

（950 题验证集，`temperature=0 / top_p=1 / max_tokens=32768`，enable_thinking=True，base=Qwen3-30B-A3B 挂载 r=32 adapter。生成用 vLLM + LoRARequest，打分用 scripts/evaluate_baseline.py。）

| category | correct | total | weightage | accuracy | contribution |
|---|---:|---:|---:|---:|---:|
| gravity | 160 | 160 | 16.8% | **100.0%** | 16.8% |
| numeral | 158 | 158 | 16.6% | **100.0%** | 16.6% |
| unit_conversion | 159 | 159 | 16.7% | **100.0%** | 16.7% |
| cipher | 155 | 157 | 16.5% | **98.7%** | 16.3% |
| equation_numeric_deduce | 21 | 60 | 6.3% | 35.0% | 2.2% |
| equation_numeric_guess | 1 | 14 | 1.5% | 7.1% | 0.1% |
| cryptarithm_deduce | 3 | 66 | 6.9% | 4.5% | 0.3% |
| bit_manipulation | 5 | 160 | 16.8% | 3.1% | 0.5% |
| cryptarithm_guess | 0 | 16 | 1.7% | 0.0% | 0.0% |
| **TOTAL** | **662** | **950** | 100.0% | **69.7%** | 69.7% |

对比 Qwen3-30B-A3B 零样本 baseline（65.3%，620/950），LoRA 净提升 **+4.4pp**：

| category | baseline | LoRA | Δ |
|---|---:|---:|---:|
| cipher | 42.7% | 98.7% | **+56.0** |
| gravity | 99.4% | 100.0% | +0.6 |
| cryptarithm_deduce | 3.0% | 4.5% | +1.5 |
| numeral | 100.0% | 100.0% | = |
| unit_conversion | 100.0% | 100.0% | = |
| cryptarithm_guess | 0.0% | 0.0% | = |
| equation_numeric_guess | 7.1% | 7.1% | = |
| equation_numeric_deduce | 53.3% | 35.0% | **-18.3** |
| bit_manipulation | 26.2% | 3.1% | **-23.1** |
| **TOTAL** | **65.3%** | **69.7%** | **+4.4** |

**关键发现**：净提升几乎全部来自 cipher（42.7%→98.7%，单类贡献 +9.2pp），但 bit_manipulation 与 equation_numeric_deduce 明显退化，两类合计丢 48 题。抽查 bit_manipulation 错题发现：LoRA 被 SFT 锁进一套固定 CoT 模板（逐位罗列 + Matching/Best 伪搜索），对 bit_manipulation 推不出规则，且 59%（94/160）产不出合法的 8 位二进制答案（位数错 / 空 boxed / 退化重复刷爆 token），属输出格式被训练带偏而非单纯答错。若这两类维持 baseline 正确率，总分可达 **74.7%（710/950）**，是当前最大的一块可回收收益。

完整 per-example 结果见 results/lora_finetuning/qwen3-30b-a3b-lora/qwen3-30b-a3b-lora_validation.csv；错题按类别分文件保存在 qwen3-30b-a3b-lora_mistakes/ 目录下。
