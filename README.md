# Nemotron Reasoning

本仓库用于 NVIDIA Nemotron Reasoning Challenge 的 LoRA 微调与本地验证，当前实验基座模型是 **Qwen3-30B-A3B**。

## 当前结论

在 950 题固定验证集上，当前 Qwen3 结果如下：

| 模型 | 设置 | 分数 |
|---|---|---:|
| Qwen3-30B-A3B | SFT 前 | **65.3%** (620/950) |
| Qwen3-30B-A3B | SFT 后 | **69.7%** (662/950) |

详细分类结果见：

- `docs/experiments/prompting_baselines.md`
- `docs/experiments/lora_finetuning.md`

## 仓库结构

```text
configs/
  eval/                                  固定验证集 id
  training/                              LoRA 训练配置
scripts/
  create_validation_split.py             构造固定验证集
  generate_baseline.py                   base model zero-shot 推理
  evaluate_adapter.py                    base + LoRA adapter 推理
  evaluate_baseline.py                   读取 raw outputs 并打分
src/
  data/                                  SFT 数据构造
  evaluation/                            答案抽取与评分
  prompting/                             验证集读取
  providers/                             vLLM 推理封装
  training/                              YAML 配置解析与采样工具
docs/experiments/                        实验记录与运行说明
data/
  train.csv                              原始训练集
  train_split_with_cot.csv               SFT 训练数据
```

## 评测约束

训练结果产出 LoRA adapter，而不是完整模型。评估时用 vLLM 加载 base model 和 adapter，并要求最终答案尽量放在 LaTeX `\boxed{}` 中。

关键参数：

```text
base model: NVIDIA Nemotron-3-Nano-30B
max_lora_rank: 32
max_tokens: 7680
top_p: 1.0
temperature: 0.0
max_num_seqs: 64
gpu_memory_utilization: 0.85
max_model_len: 8192
```

## 环境

```bash
conda create -n nemotron python=3.10 -y
conda activate nemotron
pip install -r requirements.txt
```

## 模型权重

模型权重不入库，需要自行下载到 `models/` 目录下：

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B \
  --local-dir ./models/Qwen3-30B-A3B
```

## Qwen3-30B LoRA 训练

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ACCELERATE_BYPASS_DEVICE_MAP=true \
PYTHONUNBUFFERED=1 \
python scripts/train_lora_unsloth.py \
  --config configs/training/lora_unsloth_qwen3_30b_a3b.yaml
```

## Zero-Shot 推理与评分

```bash
python scripts/generate_baseline.py \
  --model-name qwen3-30b-a3b \
  --model-path ./models/Qwen3-30B-A3B \
  --out results/prompting_baselines/qwen3-30b-a3b \
  --force

python scripts/evaluate_baseline.py \
  --run-dir results/prompting_baselines/qwen3-30b-a3b
```

## LoRA Adapter 推理与评分

```bash
python scripts/evaluate_adapter.py \
  --model-name qwen3-30b-a3b-lora \
  --base-model ./models/Qwen3-30B-A3B \
  --adapter outputs/lora_finetuning/qwen3_30b_a3b/adapter \
  --out results/lora_finetuning/qwen3-30b-a3b-lora \
  --force

python scripts/evaluate_baseline.py \
  --run-dir results/lora_finetuning/qwen3-30b-a3b-lora
```

## 开发进度

- ✅ 训练与推理环境配置。
- ✅ LoRA SFT 实验框架搭建。
- ✅ zero-shot 与 LoRA adapter 的推理评估流程。
- ✅ Qwen3-30B-A3B 固定验证集上的 SFT 前后对比。
- ⬜ 系统性的超参数搜索实验。
