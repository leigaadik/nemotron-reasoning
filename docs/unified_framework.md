# Unified Qwen LoRA Framework

## Architecture

```text
legacy external CoT CSV ─┐
                         ├─> common tokenized JSONL ─> Transformers Trainer
synthetic generator/CoT ─┘       input_ids / attention_mask / labels
                                                    |
                                                    v
                                           Qwen3 LoRA adapters
                                                    |
                                                    |
                                                    v
                                             current_950
```

Both training datasets use the same Qwen renderer, completion-only labels,
collator, sampler, optimizer, scheduler, and LoRA configuration. The only
training variable is the data source.

## Build Data

Legacy external CoT:

```bash
python scripts/build_data.py --config configs/data/legacy.yaml
```

Solver-verified synthetic CoT:

```bash
python scripts/build_data.py --config configs/data/synthetic_pilot.yaml
```

Both commands produce the same training schema under `outputs/data/`:

```text
id, source, category, prompt, gold_answer, target_answer,
derived_answer, cot_quality, cot_correct,
input_ids, attention_mask, labels,
num_tokens, num_prompt_tokens, num_loss_tokens
```

Legacy rows are marked `cot_quality=external_unverified`. Synthetic rows are
marked `cot_quality=solver_verified` and only solver-correct rows are retained.
Legacy evaluation overlap is recorded and allowed because the historical CSV
contains the original benchmark questions. Synthetic overlap is forbidden.

## Train

```bash
python scripts/train.py --config configs/training/legacy.yaml
# Copy and edit the template: cp configs/training/legacy.yaml configs/training/<experiment>.yaml
```

Both runs use:

```text
Unsloth Qwen3 model patches and LoRA injection
Transformers Trainer and TrainingArguments
Explicit completion-only labels
LoRA r=16, alpha=32, q/k/v/o projections
Effective batch size 4
Cosine LR 2e-4 -> 1e-5
One epoch, max sequence length 8192
```

Adapters are written to:

```text
outputs/adapters/legacy-cot-transformers/
outputs/adapters/synthetic-cot-transformers/
```

## Evaluate

Run all four independent evaluations:

```bash
bash scripts/run_evaluation_matrix.sh \
  ./models/Qwen3-30B-A3B \
  outputs/adapters/legacy-cot-transformers \
  outputs/adapters/synthetic-cot-transformers \
  results/evaluation_matrix/qwen3-30b
```

The matrix is:

| Adapter | current_950 |
|---|---|
| legacy-cot-transformers | independent inference |
| synthetic-cot-transformers | independent inference |
| low-quality-cot-transformers | independent inference |

Each cell has its own raw outputs, run metadata, validation CSV, summary CSV,
mistake files, inference log, and scoring log. All four cells use identical
vLLM and scoring settings.

## 实验结果

> 训练设备：**NVIDIA H200（单卡）**。评估日期：2026-08-11。

共享训练配置（三组 adapter 完全一致）：

| 参数 | 取值 |
|---|---|
| base model | Qwen3-30B-A3B |
| LoRA r / alpha / dropout | 16 / 32 / 0.0 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| learning_rate (cosine_with_min_lr) | 2e-4 → 1e-5 |
| effective batch size | 4 |
| epochs | 1 |
| max_seq_length | 8192 |

训练数据对比：

| | legacy-cot-transformers | synthetic-cot-transformers | low-quality-cot-transformers |
|---|---|---|---|
| 来源 | train_split_with_cot.csv（竞赛官方训练集） | solver-verified generator（nvidia-nemotron） | train_split_low_quality_cot.csv（消融对照） |
| 行数 | 9,500 | 13,794 | 9,500 |
| CoT 质量 | external_unverified | solver_verified | template_fabricated（极短模板，无真实推理） |
| avg_tokens / avg_loss_tokens | 3,442 / 3,299 | 3,480 / 3,328 | 202 / 59 |
| current_950 重叠 | **85%**（8,082 行 id+prompt 完全重合） | **0%**（数据集构建时明确禁止） | 未统计 |
| 训练步数 / 总时长 | 2,375 步 / 6.26h | 3,449 步 / 8.53h | 2,375 步 / 2.89h |
| train_loss | 0.005811 | 0.009022 | 0.078611 |

### TOTAL 准确率汇总

| adapter | current_950 |
|---|---:|
| legacy-cot-transformers | **86.7%** (824/950) |
| synthetic-cot-transformers | **91.2%** (866/950) |
| low-quality-cot-transformers | **70.2%** (667/950) |

### current_950 各类别明细

| category | legacy-cot-transformers | synthetic-cot-transformers | low-quality-cot-transformers |
|---|---:|---:|---:|
| bit_manipulation | 80.6% (129/160) | **96.9%** (155/160) | 41.9% (67/160) |
| cipher | 100.0% (157/157) | 100.0% (157/157) | 73.9% (116/157) |
| cryptarithm_deduce | 9.1% (6/66) | 12.1% (8/66) | 4.5% (3/66) |
| cryptarithm_guess | 0.0% (0/16) | **12.5%** (2/16) | 0.0% (0/16) |
| equation_numeric_deduce | 90.0% (54/60) | **100.0%** (60/60) | 51.7% (31/60) |
| equation_numeric_guess | 7.1% (1/14) | **50.0%** (7/14) | 14.3% (2/14) |
| gravity | 100.0% (160/160) | 100.0% (160/160) | 81.9% (131/160) |
| numeral | 100.0% (158/158) | 100.0% (158/158) | 100.0% (158/158) |
| unit_conversion | 100.0% (159/159) | 100.0% (159/159) | 100.0% (159/159) |
| **TOTAL** | **86.7% (824/950)** | **91.2% (866/950)** | **70.2% (667/950)** |
