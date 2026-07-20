# Prompting Baselines Experiment

本文档记录 `exp/prompting-baselines` 分支的开发目标和实验计划。

## 分支目标

本分支用于评估多个主流大语言模型在不进行微调的情况下，仅通过 zero-shot prompting 完成 NVIDIA Nemotron Reasoning Challenge 题目的推理能力。

实验重点不是训练 LoRA adapter，而是建立一个可复现的 prompting baseline：

- 比较不同模型厂商、不同开源/闭源模型的原始推理能力。
- 使用与官方评测一致的答案格式约束。
- 在固定验证集上评估准确率，形成可对比的结果表。
- 为后续 LoRA 微调结果提供 baseline 参照。

## 项目运作逻辑

本任务采用 **生成 / 评估两阶段分离** 的流程，保证昂贵的 GPU 推理只跑一次，而答案抽取和打分规则可以随时重跑不动模型。

### 两阶段流水线

```
data/train.csv (id, prompt, answer)
        |
        |  join by id
        v
configs/eval/validation_ids_seed42_size950.csv (id, category)
        |
        |  src/prompting/dataset.py :: load_validation
        v
    Example(id, prompt, answer, category) x 950
        |
        |  拼 prompt_suffix + tokenizer.apply_chat_template(enable_thinking=True)
        |  vllm.LLM.generate   (temperature=0, top_p=1, max_tokens=32768)
        |
        v
== 阶段 1  scripts/generate_baseline.py  (GPU, 慢) ==
   results/prompting_baselines/<model>/
       <model>_raw_outputs.jsonl     # 每题一行原始生成
       <model>_run.yaml              # 参数快照 + git commit + 时间戳
        |
        v
== 阶段 2  scripts/evaluate_baseline.py  (CPU, 秒级, 可重跑) ==
   同目录再写入:
       <model>_validation.csv        # 950 行 per-example: id/prompt/answer/output/category/predicted/correct
       <model>_results.csv           # 每类 + TOTAL: correct/total/weightage/percentage/contribution
       <model>_mistakes/<cat>.csv    # 每类错题一份
```

### 阶段 1：生成

- 加载一个本地模型权重到 vLLM（tensor_parallel_size=1，max_model_len=32768，enable_prefix_caching）。
- 对 950 题一次性 batch 送进 `llm.generate`，vLLM 内部 continuous batching。
- 每条 raw output 写一行 JSONL；**不做任何答案抽取或打分**——这样以后调整 extract 逻辑不必重跑推理。
- 进程退出前 `provider.close()` 释放显存。三个模型串行跑，一次一个进程，杜绝显存残留。
- 所有推理参数（sampling / vLLM engine / chat kwargs / prompt_suffix）硬编码在 `scripts/generate_baseline.py` 顶部，三模型共享一套，保证对比公平；每次运行会把当时的完整参数字典快照到 `<model>_run.yaml.params` 里作 provenance。

### 阶段 2：评估

- 读 `<model>_raw_outputs.jsonl`，重新 join 验证集拿 gold answer + category。
- 每条调用 `src/evaluation/scoring.py::extract_final_answer`（优先 `\boxed{...}`，兜底 `Final answer:` 等模式，最后落到最后一个数字/最后一行）。
- 每条调用 `src/evaluation/scoring.py::verify`（01 串严格比 / 可 float 化的 1e-2 相对容忍 / 其余大小写不敏感字符串比）。
- 输出格式与 `notebooks/evaluation/adapter_validation.ipynb` cell 24 **完全一致**，方便与 LoRA 阶段的产物直接对拍。

### 关键设计选择

- **验证集 ID 固定**（`configs/eval/validation_ids_seed42_size950.csv`, seed=42, size=950, 分层）——所有模型跑同一 950 题；不重跑验证集切分。
- **每模型独立进程**——不在同一 Python 进程内切换 vLLM 实例，避免 GPU 内存/句柄泄漏。
- **参数不做 per-model 定制**——config 化简后所有模型共享同一套 `max_tokens=32768 / temperature=0 / top_p=1`，若将来某模型确实需要小改，再重新引入 config 层。
- **模型权重不入库**——`models/` 与 `results/` 都在 `.gitignore`；只有代码、脚本、docs 里手工填的结果表进 git。

### 一次完整运行

```bash
PY=/opt/conda/bin/python
cd <REPO_ROOT>

# 阶段 1：三次调用，每次一个模型（GPU 串行）
$PY scripts/generate_baseline.py \
    --model-name nemotron-3-nano-30b-a3b \
    --model-path <REPO_ROOT>/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --out results/prompting_baselines/nemotron-3-nano-30b-a3b

$PY scripts/generate_baseline.py \
    --model-name qwen3-30b-a3b \
    --model-path <REPO_ROOT>/models/Qwen3-30B-A3B \
    --out results/prompting_baselines/qwen3-30b-a3b

$PY scripts/generate_baseline.py \
    --model-name glm-4.7-flash \
    --model-path <REPO_ROOT>/models/GLM-4.7-Flash \
    --out results/prompting_baselines/glm-4.7-flash

# 阶段 2：一次评估三个 run-dir
$PY scripts/evaluate_baseline.py \
    --run-dir results/prompting_baselines/nemotron-3-nano-30b-a3b \
    --run-dir results/prompting_baselines/qwen3-30b-a3b \
    --run-dir results/prompting_baselines/glm-4.7-flash
```

阶段 2 会在每个 run-dir 里打印一张类似 `adapter_validation.ipynb` cell 24 的表：

```
              correct  total  weightage  percentage  contribution
bit_manipulation   ...    ...      ...        ...%          ...%
cipher             ...    ...      ...        ...%          ...%
...
TOTAL              ...    950   100.0%       ...%          ...%
```

TOTAL 行的 `percentage` 即该模型 zero-shot 准确率；填入本文档 `## 目标结果格式` 表格即完成一轮对比实验。

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

## 代码结构与实现现状

分支 `exp/prompting-baselines` 上与本任务直接相关的文件：

```
docs/experiments/prompting_baselines.md      本文档：分支目标 + 实验计划 + 运作逻辑
configs/eval/
  validation_ids_seed42_size950.csv         固定验证集 ID（stratified, seed=42）
scripts/
  create_validation_split.py                生成验证集 ID
  generate_baseline.py                      阶段 1：单模型生成 raw outputs
  evaluate_baseline.py                      阶段 2：读 raw outputs -> validation/results/mistakes
src/
  evaluation/
    scoring.py                              extract_final_answer / verify / score_predictions
    categories.py                           detect_category（兜底类别判定）
  prompting/
    dataset.py                              Example dataclass + load_validation()
  providers/
    base.py                                 Provider Protocol + build_provider dispatch
    local_vllm.py                           本地 vLLM 后端（enable_thinking 自动兜底）
notebooks/evaluation/
  adapter_validation.ipynb                  LoRA 验证参考（我们的 evaluate 输出格式与 cell 24 一致）
  nvidia_nemotron_metric.ipynb              官方 metric 参考
models/                                     本地权重（gitignored, 不入库）
results/prompting_baselines/<model>/        每模型一子目录（gitignored, 不入库）
```

### 各步落地状态

1. **固定验证集**：DONE（`scripts/create_validation_split.py` 生成，`configs/eval/validation_ids_seed42_size950.csv` 入库）。
2. **统一评估函数**：DONE（`src/evaluation/scoring.py::extract_final_answer / verify / score_predictions` 复刻自 `notebooks/evaluation/nvidia_nemotron_metric.ipynb`）。
3. **统一模型调用接口**：PARTIAL——第一版只落地 `src/providers/local_vllm.py`（本地 vLLM）。当前不需要 `openai_compatible.py / transformers_local.py`，等真正接 API 或 HF 推理时再补，接口层 `src/providers/base.py` 已经预留。
4. **prompting baseline runner**：DONE，拆成 `scripts/generate_baseline.py`（阶段 1）+ `scripts/evaluate_baseline.py`（阶段 2），产物结构与 `adapter_validation.ipynb` cell 24 对齐（`<model>_validation.csv / <model>_results.csv / <model>_mistakes/<cat>.csv`）。
5. **多模型结果汇总**：DEFERRED——本轮不做独立 summarize 脚本；三个模型的 `<model>_results.csv` 里 TOTAL + per-category 已足够手工填进本文档 `## 目标结果格式` 表格。若模型数继续增长再考虑重新引入。
6. **对比 LoRA 结果**：DEFERRED——等 LoRA adapter 训练完成后再做，本分支 baseline 即为对照组。

## 分支开发原则

- 不在本分支提交模型权重、adapter、checkpoint 或大规模结果文件（`models/`、`results/` 均已 gitignore）。
- 每次实验必须记录模型名称、参数、验证集版本和 prompt 模板——`generate_baseline.py` 每次运行都会把参数快照到 `<model>_run.yaml` 里。
- Prompt 模板变更应单独记录，避免结果不可比较——修改 `scripts/generate_baseline.py::PROMPT_SUFFIX` 时请单独 commit。
- 默认使用确定性生成参数，除非实验明确需要采样。

默认参数（在 `scripts/generate_baseline.py` 顶部硬编码，所有 baseline 模型共享）：

```text
temperature: 0.0
top_p:       1.0
max_tokens:  32768
max_model_len: 32768
enable_thinking: True   （tokenizer 不支持时会自动回退）
prompt_suffix:
  Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

## 当前 Runbook

代码基础设施已就绪，接下来只需按顺序执行：

1. **跑第一个 baseline（推荐 Nemotron 或 Qwen3）**：单模型全量 950 题，验证 vLLM 环境无异常、`finish_reason` 里 `length` 占比合理（说明 32k 生成够用）、`\boxed{}` 命中率不异常低。
2. **看 mistakes**：翻 `results/prompting_baselines/<model>/<model>_mistakes/<category>.csv` 里若干条，确认错在推理本身而不是答案抽取失败。
3. **跑剩下两个模型**：GLM-4.7-Flash 需要等权重下载完（曾经缺 `tokenizer_config.json / tokenizer.json`，运行前先 `ls models/GLM-4.7-Flash | grep tokenizer` 确认）。
4. **一次性 evaluate 三个 run-dir**：拿到三张 `<model>_results.csv`。
5. **手工填结果表进本文档 `## 目标结果格式`**：`model | think | max_tokens | score`，score 取 `_results.csv` 的 TOTAL 行 percentage。
6. **Commit**：代码基础设施 + docs 里补的结果表；`results/` 已被 gitignore，不会误提。

后续如果要扩展到 API 模型（DeepSeek / Claude / Gemini），只需新增 `src/providers/openai_compatible.py` 并在 `src/providers/base.py::build_provider` 里注册；`scripts/generate_baseline.py` 的 CLI 层可能需要加一个 `--provider` 开关，其它逻辑不动。
