# Nemotron Reasoning

本仓库用于 NVIDIA Nemotron Reasoning Challenge。

项目目标是通过 LoRA 微调大语言模型，提升模型在 reasoning benchmark 上的准确率。当前仓库主要保存训练数据集，后续可以在此基础上补充训练脚本、评估脚本和实验配置。

## 仓库内容

```text
data/
  train.csv
  train_split_with_cot.csv
notebooks/
  unsloth_lora_training.ipynb
```

- `data/train.csv`：比赛原始训练数据。
- `data/train_split_with_cot.csv`：基于原始数据清洗、整理并构造出的高质量 chain-of-thought 训练数据。
- `notebooks/unsloth_lora_training.ipynb`：基于 Unsloth 的 LoRA 训练 notebook。

## 比赛数据集

比赛数据集由一组逻辑推理谜题组成，目标是识别并应用隐藏的转换规则。题目覆盖多个推理领域，例如 bit manipulation、代数方程等。

本仓库当前保留的原始比赛数据为：

```text
train.csv
```

`train.csv` 是原始训练集，包含 puzzle 及其标准答案：

- `id`：每道题的唯一标识。
- `prompt`：题目描述，包含输入输出示例以及需要求解的具体实例。
- `answer`：该题的标准答案。

此外，比赛页面会提供 `test.csv` 作为编写提交流程的示例测试集；正式评分时会被替换为包含数百道题目的隐藏测试集。该文件不在本仓库当前数据目录中。

- `id`：每道题的唯一标识。
- `prompt`：题目描述，格式与 `train.csv` 一致。

本仓库中的 `data/train_split_with_cot.csv` 是基于 `train.csv` 进一步清洗、筛选和构造得到的高质量训练数据，用于 LoRA 微调实验。

原始训练数据集信息：

```text
文件：train.csv
大小：约 3.07 MB
格式：CSV
许可证：CC BY 4.0
```

## LoRA 提交要求

比赛最终提交物不是完整模型，而是一个 LoRA adapter 压缩包：

```text
submission.zip
```

核心要求：

- base model：`NVIDIA Nemotron-3-Nano-30B`
- 提交物必须是兼容该 base model 的 LoRA adapter。
- LoRA rank 最大为 `32`。
- adapter 中必须包含 `adapter_config.json`。
- 评测时官方会使用 vLLM 加载 base model 和提交的 LoRA adapter。

## 官方评测设置

官方评测会让模型生成答案，并要求最终答案放在 LaTeX 的 `\boxed{}` 中。评分脚本会优先提取 `\boxed{}` 内的内容，如果没有找到，则退回到其他启发式规则或最后一个数值。

评测参数：

```text
max_lora_rank: 32
max_tokens: 7680
top_p: 1.0
temperature: 0.0
max_num_seqs: 64
gpu_memory_utilization: 0.85
max_model_len: 8192
```

这意味着训练和验证时应尽量关注：

- LoRA rank 不超过 32。
- 适配 8K 上下文长度。
- 输出格式稳定包含 `\boxed{final answer}`。
- adapter 能被 vLLM 正常加载。

## 官方硬件环境

比赛说明中提到，挑战赛计算资源由 Google Cloud 提供，使用 G4 VMs，底层 GPU 为：

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

该环境适合 Nemotron 模型的轻量微调和高吞吐推理。对于本项目而言，可以把本地和云端硬件分工理解为：

- 本地 RTX A6000：适合数据处理、训练脚本调试、小规模 LoRA/QLoRA 实验。
- 官方 G4 / RTX PRO 6000 Blackwell：更接近比赛推荐环境，适合正式复现和提交前验证。
- H200 / B200：如果可用，适合更大 batch、更长上下文和更多超参数 sweep。

## Quick Start

克隆仓库：

```bash
git clone git@github.com:leigaadik/nemotron-reasoning.git
cd nemotron-reasoning
```

创建并激活 Conda 虚拟环境：

```bash
conda create -n nemotron python=3.12 -y
conda activate nemotron
```

检查 Python 版本：

```bash
python --version
```

期望版本：

```text
Python 3.12.x
```

## 项目思路

本项目计划采用 LoRA 进行参数高效微调。基本流程如下：

1. 从 `data/` 目录读取并清洗训练数据。
2. 选择合适的 reasoning base model。
3. 使用 LoRA 进行微调。
4. 在 benchmark 上评估微调后模型的准确率。
5. 根据评估结果迭代数据格式、prompt 模板和训练参数。

临时脚本、notebook、checkpoint 和本地实验输出建议先放在本地临时目录中，确认稳定后再纳入版本管理。
