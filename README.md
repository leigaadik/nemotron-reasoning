# Nemotron Reasoning

本仓库用于 NVIDIA Nemotron Reasoning Challenge。

项目目标是通过 LoRA 微调大语言模型，提升模型在 reasoning benchmark 上的准确率。当前仓库主要保存训练数据集，后续可以在此基础上补充训练脚本、评估脚本和实验配置。

## 仓库内容

```text
data/
  train.csv
  train_split_with_cot.csv
```

- `train.csv`：原始训练数据。
- `train_split_with_cot.csv`：带有 chain-of-thought 推理过程的数据版本。

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
