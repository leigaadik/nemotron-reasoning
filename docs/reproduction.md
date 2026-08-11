# 实验复现与运行流程

本文档记录从数据准备、模型推理、LoRA 训练到结果评分的完整复现实验流程，并统一使用评估集 `current_950`。

## 0. 实验环境配置

进入项目目录，创建 Python 3.10 conda 环境并激活：

```bash
cd ./nemotron-reasoning/
conda create -n nemotron python=3.10 -y
conda activate nemotron
```

安装运行依赖：

```bash
pip install -r requirements.txt
```

本地模型权重需要放在：

```text
models/Qwen3-30B-A3B/
```

可使用 Hugging Face CLI 下载到该路径：

```bash
hf download Qwen/Qwen3-30B-A3B \
  --local-dir models/Qwen3-30B-A3B
```

## 1. 零样本推理与评估

零样本推理不加载 LoRA Adapter，直接使用本地基础模型生成答案。

评估集加载方式：`configs/eval/current_950.yaml` 定义评估集名称、题目 CSV 和样本 ID 列表。程序先读取 `data/train.csv`，再根据 `configs/eval/validation_ids_seed42_size950.csv` 选取 950 条样本。

```text
configs/eval/current_950.yaml
  questions_csv: data/train.csv
  ids_csv: configs/eval/validation_ids_seed42_size950.csv
  expected_size: 950
```

推理结果写入 `results/zeroshot/qwen3-30b/current_950/`：

```bash
python scripts/infer.py \
  --run-name qwen3-30b-zeroshot \
  --base-model ./models/Qwen3-30B-A3B \
  --suite configs/eval/current_950.yaml \
  --out results/zeroshot/qwen3-30b/current_950 \
  --force
```

评分阶段会重新加载 `configs/eval/current_950.yaml` 中定义的标准答案，并读取上一步生成的原始推理结果 JSONL：

```bash
python scripts/score.py \
  --suite configs/eval/current_950.yaml \
  --raw-outputs results/zeroshot/qwen3-30b/current_950/qwen3-30b-zeroshot__current_950_raw_outputs.jsonl \
  --out results/zeroshot/qwen3-30b/current_950
```

主要产物均写入 `results/zeroshot/qwen3-30b/current_950/`：

| 文件 | 生成阶段 | 内容 |
|---|---|---|
| `qwen3-30b-zeroshot__current_950_raw_outputs.jsonl` | 推理 | 每道题的原始模型输出、生成结束原因和输出 token 数 |
| `qwen3-30b-zeroshot__current_950_run.yaml` | 推理 | 本次运行的模型路径、评估集、时间戳、git commit 和推理参数 |
| `current_950_validation.csv` | 评分 | 逐样本评分结果，包括标准答案、模型输出、抽取答案、类别和是否正确 |
| `current_950_summary.csv` | 评分 | 按类别统计的正确数、总数和准确率，并包含 `TOTAL` 总体结果 |
| `current_950_mistakes/*.csv` | 评分 | 按类别拆分的错误样本，用于误差分析 |

推理参数写在 `src/inference/runner.py`：

```text
SAMPLING = temperature 0.0, top_p 1.0, max_tokens 7680
VLLM = max_model_len 8192, gpu_memory_utilization 0.85, max_lora_rank 32
PROMPT_SUFFIX = Please put your final answer inside \boxed{}
```

## 2. 训练数据预处理与 JSONL 构建

本节生成三组 LoRA 训练所需的已分词 JSONL 数据。

### 2.1 legacy CoT 数据

读取 `data/train_split_with_cot.csv`，生成 legacy LoRA 训练数据保存到 `outputs/data/legacy/`。

```bash
python scripts/build_data.py --config configs/data/legacy.yaml
```

| 文件 | 内容 |
|---|---|
| `outputs/data/legacy/qwen_traces.jsonl` | legacy LoRA 训练数据 |
| `outputs/data/legacy/qwen_traces.validation.json` | 数据校验报告 |
| `outputs/data/legacy/qwen_traces.manifest.json` | 数据构建记录，包括配置、输入文件 hash 和统计信息 |

### 2.2 synthetic pilot 数据

读取 `src/generators/nemotron/` 中的数据生成代码，生成 synthetic LoRA 训练数据保存到 `outputs/data/synthetic_pilot/`。

```bash
python scripts/build_data.py --config configs/data/synthetic_pilot.yaml
```

| 文件 | 内容 |
|---|---|
| `outputs/data/synthetic_pilot/problems.jsonl` | 生成器产生的原始题目 |
| `outputs/data/synthetic_pilot/cot.jsonl` | solver 生成并验证后的 CoT 数据 |
| `outputs/data/synthetic_pilot/qwen_traces.jsonl` | synthetic LoRA 训练数据 |
| `outputs/data/synthetic_pilot/qwen_traces.validation.json` | 数据校验报告 |
| `outputs/data/synthetic_pilot/qwen_traces.manifest.json` | 数据构建记录，包括配置、输入文件 hash 和统计信息 |

### 2.3 low-quality CoT 数据

读取 `data/train.csv`，生成模板化 CoT CSV 到 `data/train_split_low_quality_cot.csv`，再生成 low-quality LoRA 训练数据到 `outputs/data/low_quality/`。

```bash
python scripts/build_low_quality_cot_csv.py \
  --input-csv data/train.csv \
  --output-csv data/train_split_low_quality_cot.csv \
  --n 9500 \
  --seed 42
```

然后使用 `configs/data/low_quality.yaml` 构建训练 JSONL：

```bash
python scripts/build_data.py --config configs/data/low_quality.yaml
```

输出路径：

| 文件 | 内容 |
|---|---|
| `outputs/data/low_quality/qwen_traces.jsonl` | low-quality LoRA 训练数据 |
| `outputs/data/low_quality/qwen_traces.validation.json` | 数据校验报告 |
| `outputs/data/low_quality/qwen_traces.manifest.json` | 数据构建记录，包括配置、输入文件 hash 和统计信息 |

三组 LoRA 训练最终读取的文件分别为：

| LoRA 组 | 训练 JSONL |
|---|---|
| `legacy-cot-transformers` | `outputs/data/legacy/qwen_traces.jsonl` |
| `synthetic-cot-transformers` | `outputs/data/synthetic_pilot/qwen_traces.jsonl` |
| `low-quality-cot-transformers` | `outputs/data/low_quality/qwen_traces.jsonl` |

JSONL 每行包含 `input_ids`、`attention_mask`、`labels`、`prompt`、`gold_answer`、`category` 等字段。

## 3. LoRA 训练

训练入口：

```bash
python scripts/train.py --config configs/training/<experiment>.yaml
```

三组训练配置均继承 `configs/training/base.yaml`：

```text
configs/training/legacy.yaml
configs/training/synthetic.yaml
configs/training/low_quality.yaml
```

### 3.1 legacy LoRA

配置：

```text
configs/training/legacy.yaml
```

关键输入输出：

```text
input:  outputs/data/legacy/qwen_traces.jsonl
output: outputs/adapters/legacy-cot-transformers/
log:    outputs/trainer/legacy-cot-transformers/
```

训练命令：

```bash
python scripts/train.py --config configs/training/legacy.yaml
```

### 3.2 synthetic LoRA

配置：

```text
configs/training/synthetic.yaml
```

关键输入输出：

```text
input:  outputs/data/synthetic_pilot/qwen_traces.jsonl
output: outputs/adapters/synthetic-cot-transformers/
log:    outputs/trainer/synthetic-cot-transformers/
```

训练命令：

```bash
python scripts/train.py --config configs/training/synthetic.yaml
```

### 3.3 low-quality LoRA

配置：

```text
configs/training/low_quality.yaml
```

关键输入输出：

```text
input:  outputs/data/low_quality/qwen_traces.jsonl
output: outputs/adapters/low-quality-cot-transformers/
log:    outputs/trainer/low-quality-cot-transformers/
```

训练命令：

```bash
python scripts/train.py --config configs/training/low_quality.yaml
```

三组训练完成后应得到：

```text
outputs/adapters/legacy-cot-transformers/adapter_config.json
outputs/adapters/synthetic-cot-transformers/adapter_config.json
outputs/adapters/low-quality-cot-transformers/adapter_config.json
```

## 4. LoRA 推理

三组 LoRA Adapter 分别调用 `scripts/infer.py` 进行推理。以下以 `legacy-cot-transformers` 为例；其他 LoRA Adapter 替换 `--run-name`、`--adapter` 和 `--out`。

```bash
python scripts/infer.py \
  --run-name legacy-cot-transformers \
  --base-model ./models/Qwen3-30B-A3B \
  --adapter outputs/adapters/legacy-cot-transformers \
  --suite configs/eval/current_950.yaml \
  --out results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950 \
  --force
```

三组 LoRA Adapter 的推理命令参数：

| LoRA Adapter | `--run-name` | `--adapter` | `--out` |
|---|---|---|---|
| legacy | `legacy-cot-transformers` | `outputs/adapters/legacy-cot-transformers` | `results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950` |
| synthetic | `synthetic-cot-transformers` | `outputs/adapters/synthetic-cot-transformers` | `results/adapter_eval/qwen3-30b/synthetic-cot-transformers/current_950` |
| low-quality | `low-quality-cot-transformers` | `outputs/adapters/low-quality-cot-transformers` | `results/adapter_eval/qwen3-30b/low-quality-cot-transformers/current_950` |

推理输出：

| 文件 | 内容 |
|---|---|
| `<run-name>__current_950_raw_outputs.jsonl` | 每道题的原始模型输出、生成结束原因和输出 token 数 |
| `<run-name>__current_950_run.yaml` | 本次运行的模型路径、LoRA Adapter 路径、评估集、时间戳、git commit 和推理参数 |

## 5. 结果评分

三组 LoRA Adapter 分别调用 `scripts/score.py` 进行评分。以下以 `legacy-cot-transformers` 为例；其他 LoRA Adapter 替换 `--raw-outputs` 和 `--out`。

```bash
python scripts/score.py \
  --suite configs/eval/current_950.yaml \
  --raw-outputs results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950/legacy-cot-transformers__current_950_raw_outputs.jsonl \
  --out results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950
```

三组 LoRA Adapter 的评分命令参数：

| LoRA Adapter | `--raw-outputs` | `--out` |
|---|---|---|
| legacy | `results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950/legacy-cot-transformers__current_950_raw_outputs.jsonl` | `results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950` |
| synthetic | `results/adapter_eval/qwen3-30b/synthetic-cot-transformers/current_950/synthetic-cot-transformers__current_950_raw_outputs.jsonl` | `results/adapter_eval/qwen3-30b/synthetic-cot-transformers/current_950` |
| low-quality | `results/adapter_eval/qwen3-30b/low-quality-cot-transformers/current_950/low-quality-cot-transformers__current_950_raw_outputs.jsonl` | `results/adapter_eval/qwen3-30b/low-quality-cot-transformers/current_950` |

所有评分命令均使用同一个评估集配置：

```text
--suite configs/eval/current_950.yaml
```

评分输出：

```text
results/adapter_eval/qwen3-30b/<adapter>/current_950/current_950_validation.csv
results/adapter_eval/qwen3-30b/<adapter>/current_950/current_950_summary.csv
results/adapter_eval/qwen3-30b/<adapter>/current_950/current_950_mistakes/*.csv
results/adapter_eval/qwen3-30b/<adapter>/current_950/scoring.log
```

## 6. 超参数搜索

每个超参数实验使用独立的训练配置文件和独立输出目录。以下示例基于 legacy 数据创建一个 `legacy_ml7680_lr1e-4_r32` 实验。

### 6.1 创建实验配置

复制已有 legacy 配置：

```bash
cp configs/training/legacy.yaml configs/training/legacy_ml7680_lr1e-4_r32.yaml
```

然后修改 `configs/training/legacy_ml7680_lr1e-4_r32.yaml`：

```yaml
extends: base.yaml

experiment:
  name: legacy_ml7680_lr1e-4_r32
  description: Qwen3-30B-A3B LoRA SFT on legacy CoT, max_length 7680, lr 1e-4, r 32

paths:
  train_jsonl: outputs/data/legacy_ml7680/qwen_traces.jsonl
  adapter_dir: outputs/adapters/legacy_ml7680_lr1e-4_r32

lora:
  r: 32
  alpha: 32

training:
  output_dir: outputs/trainer/legacy_ml7680_lr1e-4_r32
  learning_rate: 1.0e-4
```

未写入该文件的字段继续继承 `configs/training/base.yaml`。仅修改 learning rate、LoRA rank、batch size 等训练参数时，可以复用已有 JSONL；修改训练序列长度时，需要同步修改数据配置并重新构建训练 JSONL。

### 6.2 修改数据长度

若实验需要 `max_length=7680`，复制并修改数据配置：

```bash
cp configs/data/legacy.yaml configs/data/legacy_ml7680.yaml
```

修改 `configs/data/legacy_ml7680.yaml`：

```yaml
data:
  traces_jsonl: outputs/data/legacy_ml7680/qwen_traces.jsonl
  max_length: 7680
```

重新构建训练数据：

```bash
python scripts/build_data.py --config configs/data/legacy_ml7680.yaml
```

产物：

```text
outputs/data/legacy_ml7680/qwen_traces.jsonl
outputs/data/legacy_ml7680/qwen_traces.validation.json
outputs/data/legacy_ml7680/qwen_traces.manifest.json
```

### 6.3 启动训练

```bash
python scripts/train.py --config configs/training/legacy_ml7680_lr1e-4_r32.yaml
```

训练产物：

```text
outputs/adapters/legacy_ml7680_lr1e-4_r32/
outputs/trainer/legacy_ml7680_lr1e-4_r32/
```

### 6.4 推理与评分

```bash
python scripts/infer.py \
  --run-name legacy_ml7680_lr1e-4_r32 \
  --base-model ./models/Qwen3-30B-A3B \
  --adapter outputs/adapters/legacy_ml7680_lr1e-4_r32 \
  --suite configs/eval/current_950.yaml \
  --out results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950 \
  --force

python scripts/score.py \
  --suite configs/eval/current_950.yaml \
  --raw-outputs results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950/legacy_ml7680_lr1e-4_r32__current_950_raw_outputs.jsonl \
  --out results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950
```

评估产物：

```text
results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950/current_950_summary.csv
results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950/current_950_validation.csv
results/hparam_search/legacy_ml7680_lr1e-4_r32/current_950/current_950_mistakes/*.csv
```

### 6.5 常用搜索字段

| 目标 | 配置字段 |
|---|---|
| 训练序列长度 | `configs/data/<experiment>.yaml` 中的 `data.max_length`，以及训练配置中的 `paths.train_jsonl` |
| LoRA rank | `lora.r` |
| LoRA alpha | `lora.alpha` |
| LoRA dropout | `lora.dropout` |
| LoRA 注入模块 | `lora.target_modules` |
| 学习率 | `training.learning_rate` |
| 最小学习率 | `training.lr_scheduler_kwargs.min_lr` |
| warmup | `training.warmup_ratio` |
| batch size | `training.per_device_train_batch_size` 和 `training.gradient_accumulation_steps` |
| 训练轮数 | `training.num_train_epochs` |

每个实验必须使用唯一的 `experiment.name`、`paths.adapter_dir`、`training.output_dir` 和评估输出目录，避免覆盖已有结果。
