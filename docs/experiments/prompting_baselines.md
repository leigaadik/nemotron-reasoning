# Prompting Baselines Experiment

本文档记录 `exp/prompting-baselines` 分支的开发目标和实验计划。

## 分支目标

本分支用于评估多个主流大语言模型在不进行微调的情况下，仅通过 zero-shot prompting 完成 NVIDIA Nemotron Reasoning Challenge 题目的推理能力。

实验重点不是训练 LoRA adapter，而是建立一个可复现的 prompting baseline：

- 比较不同模型厂商、不同开源/闭源模型的原始推理能力。
- 使用与官方评测一致的答案格式约束。
- 在固定验证集上评估准确率，形成可对比的结果表。
- 为后续 LoRA 微调结果提供 baseline 参照。

## Prompt 协议

所有模型默认使用与官方评测一致的答案格式提示：

```python
prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
```

评估时只比较最终答案，优先提取 `\boxed{}` 中的内容。

## 目标结果格式

最终希望得到类似下面的结果表：

| model | think | max_tokens | score |
|---|---:|---:|---:|
| Gemini-3.1-Pro | yes | 32768 | 0.81 |
| Claude-Opus-4.6 | yes | 32768 | 0.78 |
| DeepSeek-V3.2 | yes | 32768 | 0.74 |
| Qwen3-Max | yes | 32768 | 0.72 |
| Claude-Sonnet-4.5 | no | 32768 | 0.51 |

表格字段含义：

- `model`：被评估模型名称。
- `think`：是否启用模型的 thinking/reasoning 模式。
- `max_tokens`：生成上限。
- `score`：在验证集上的准确率。

后续可以扩展字段：

- `provider`：模型提供方。
- `temperature`：采样温度。
- `top_p`：top-p 参数。
- `eval_size`：评估样本数量。
- `latency_avg`：平均响应耗时。
- `cost_estimate`：估算调用成本。
- `notes`：模型异常、格式失败、限流等备注。

## 候选模型选择

比赛官方目标模型是 `NVIDIA Nemotron-3-Nano-30B-A3B`。从公开技术资料看，它不是普通 dense Transformer，而是 **Mixture-of-Experts hybrid Mamba-Transformer** 模型；`30B-A3B` 表示总参数约 30B 量级、每次推理激活约 3B 量级参数。

因此，本分支选择 baseline 模型时优先考虑：

1. MoE 或 sparse/MoE-like 模型。
2. 总参数接近 20B-40B，或 active 参数接近 A3B。
3. 支持长上下文和 reasoning/thinking 模式。
4. 能通过 API、vLLM、Transformers 或 OpenAI-compatible server 稳定调用。

### 第一优先级：结构和规模更接近的 MoE 模型

| model | family/provider | structure | why test |
|---|---|---|---|
| Nemotron-3-Nano-30B-A3B | NVIDIA | MoE hybrid Mamba-Transformer | 官方 baseline/base model，必须作为核心参照 |
| Qwen3-30B-A3B | Alibaba/Qwen | MoE, A3B | 与 Nemotron 的 30B-A3B 命名和激活规模最接近，是最重要的横向对比 |
| Qwen3-30B-A3B-Thinking | Alibaba/Qwen | MoE, A3B, thinking | 适合测试 thinking mode 对逻辑推理题的增益 |
| GPT-OSS-20B | OpenAI | MoE/open-weight | 总规模略小，但同属 sparse/MoE 路线，可作为近邻开源 baseline |
| GLM-4.7-Flash | Z.ai/GLM | MoE, 30B-A3B if available | 如果可用，参数标注与 Nemotron/Qwen3 的 30B-A3B 非常接近，适合作为国内 MoE 对照 |
| Mixtral-8x7B-Instruct | Mistral | MoE | 较早的开源 MoE，对照价值高，但架构和训练代际较旧 |
| Llama-4-Scout | Meta | MoE | Llama 4 系列改用 MoE，可作为 Meta 阵营的高效 MoE 对照 |
| Llama-4-Maverick | Meta | MoE | 更强的 Llama 4 MoE 版本，适合作为高性能参考组 |

### 第二优先级：30B 附近 dense/open-weight 对照组

这些模型不是 MoE，但参数规模接近，适合回答一个问题：**MoE 结构本身是否在本任务上带来优势，还是主要由模型训练和推理能力决定。**

| model | family/provider | structure | why test |
|---|---|---|---|
| Qwen2.5-32B-Instruct | Alibaba/Qwen | dense Transformer | 30B 附近强 dense baseline |
| Qwen3-32B | Alibaba/Qwen | dense Transformer | 与 Qwen3-30B-A3B 同家族，可比较 dense vs MoE |
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek/Qwen | dense distilled reasoning model | 推理能力强，适合验证 reasoning distillation 的效果 |
| Gemma-3-27B-it | Google | dense Transformer | 规模接近，作为 Google 开源模型对照 |
| Gemma-4-26B-A4B | Google | MoE, A4B | 如果可用，参数规模和 active 参数都接近 Nemotron-3-Nano-30B-A3B，是 Gemma 系列里更合适的结构对照 |
| Gemma-4-E4B | Google | efficient/MoE-like | 更轻量，适合低成本 sanity check，但不应和 30B-A3B 直接等价比较 |
| Mistral-Small-24B-Instruct | Mistral | dense Transformer | 24B 级别轻量 dense baseline |
| Yi-1.5-34B-Chat | 01.AI | dense Transformer | 34B 级别中文/英文综合能力对照 |

### GLM、Gemma、Llama 的选择建议

如果专门考虑 GLM、Gemma、Llama 三条模型线，可以按下面优先级选择：

| family | first choice | role | note |
|---|---|---|---|
| GLM | GLM-4.7-Flash | 近结构 MoE 对照 | 如果确认为 30B-A3B，则非常适合和 Nemotron-3-Nano-30B-A3B、Qwen3-30B-A3B 放在同一组 |
| Gemma | Gemma-4-26B-A4B | 近规模 MoE 对照 | 比 Gemma-3-27B-it 更贴近 MoE/A3B-A4B 设定 |
| Gemma | Gemma-3-27B-it | dense 对照 | 如果 Gemma 4 权重或推理服务不可用，可以先用它作为 Google dense baseline |
| Llama | Llama-4-Scout | MoE 对照 | 更偏高效/长上下文，适合测试 Meta MoE 系列的 zero-shot 表现 |
| Llama | Llama-4-Maverick | 强模型参考 | 若资源/API 可用，适合作为高性能对照；但规模和能力可能强于本项目目标模型 |

不建议第一轮优先使用 Llama 3.x 70B 作为主对照，因为它是 dense 且规模显著更大；它可以作为“强 dense 上限参考”，但不适合作为 Nemotron-3-Nano-30B-A3B 的结构近邻。

### 第三优先级：更大模型或闭源 API 参考组

这些模型不一定结构或规模接近，但可以作为“当前主流 LLM zero-shot prompting 上限”的参考。

| model | type | why test |
|---|---|---|
| DeepSeek-V3.2 | API / large MoE | 强 reasoning API baseline |
| Claude-Sonnet / Claude-Opus | API | 闭源强推理模型参照 |
| Gemini Pro | API | 长上下文和推理能力参照 |
| Kimi-K / Moonshot | API | 中文环境和长上下文模型参照 |
| GLM 系列 | API/open-weight depending on version | 国内主流模型参照 |

### 推荐首批实验顺序

为了避免一开始评估范围过大，建议先跑一个小而有解释力的组合：

1. `Nemotron-3-Nano-30B-A3B`：官方目标模型，不加 LoRA。
2. `Qwen3-30B-A3B`：最接近的 MoE/A3B 对照。
3. `Qwen3-30B-A3B-Thinking`：测试 thinking mode。
4. `GLM-4.7-Flash`：如果可用，作为另一组 30B-A3B MoE 对照。
5. `Gemma-4-26B-A4B`：如果可用，作为 Gemma MoE/A4B 对照。
6. `Qwen2.5-32B-Instruct` 或 `Qwen3-32B`：30B dense 对照。
7. `DeepSeek-R1-Distill-Qwen-32B`：reasoning-distilled dense 对照。

首批实验完成后，再扩展到更多 API 模型和更大模型。

## 开发任务拆解

### 1. 固定验证集

从 `data/train.csv` 中构造一个固定 validation subset，避免每次实验样本不同导致结果不可比。

建议方案：

- 使用固定随机种子。
- 默认抽样 950 条，和现有 `adapter_validation.ipynb` 的规模保持一致。
- 保存验证集 ID 列表，而不是重复保存完整数据。

建议产物：

```text
configs/eval/validation_ids_seed42_size950.csv
```

### 2. 统一评估函数

复用官方 metric notebook 中的核心逻辑：

- `extract_final_answer`
- `verify`
- 分类统计逻辑

建议沉淀为脚本：

```text
src/evaluation/extract_answer.py
src/evaluation/scoring.py
```

### 3. 统一模型调用接口

为不同模型提供方建立统一接口，避免每个模型写一套评估代码。

建议先支持本地/open-weight 模型，再支持 API 模型：

```text
src/providers/
  base.py
  openai_compatible.py
  local_vllm.py
  transformers_local.py
```

### 4. 实现 prompting baseline runner

runner 负责：

1. 读取验证集。
2. 构造统一 prompt。
3. 调用指定模型。
4. 保存原始输出。
5. 提取最终答案。
6. 计算准确率和分类准确率。

建议产物：

```text
scripts/run_prompting_baseline.py
```

输出目录：

```text
results/prompting_baselines/
  <model_name>/
    raw_outputs.jsonl
    predictions.csv
    metrics.json
    mistakes.csv
```

注意：`results/` 默认不提交到 Git，稳定结果可以整理为 Markdown/CSV 摘要后再提交。

### 5. 结果汇总

将不同模型的 `metrics.json` 汇总成统一表格：

```text
reports/prompting_baselines.md
reports/prompting_baselines.csv
```

### 6. 对比 LoRA 结果

当 LoRA adapter 训练完成后，将 prompting baseline 作为对照组：

- zero-shot prompting baseline
- LoRA fine-tuned model
- LoRA + improved prompting

这样可以判断准确率提升来自模型本身、prompt，还是 LoRA 微调。

## 分支开发原则

- 不在本分支提交模型权重、adapter、checkpoint 或大规模结果文件。
- 每次实验必须记录模型名称、参数、验证集版本和 prompt 模板。
- Prompt 模板变更应单独记录，避免结果不可比较。
- 默认使用确定性生成参数，除非实验明确需要采样。

建议默认参数：

```text
temperature: 0.0
top_p: 1.0
max_tokens: 32768 或模型允许的最大合理值
```

## 当前优先级

1. 创建固定验证集 ID。
2. 抽出 `extract_final_answer` 和 `verify`。
3. 写一个最小可运行的 baseline runner。
4. 先跑 1-2 个本地或 API 模型，验证流程。
5. 再扩展到多模型对比表。
