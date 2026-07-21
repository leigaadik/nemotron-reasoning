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
== 阶段 1  scripts/generate_baseline.py ==
   results/prompting_baselines/<model>/
       <model>_raw_outputs.jsonl     # 每题一行原始生成
       <model>_run.yaml              # 参数快照 + git commit + 时间戳
        |
        v
== 阶段 2  scripts/evaluate_baseline.py ==
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

### 环境兼容性说明

若引擎启动时报出如下 `ImportError`：

```
ImportError: .../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

说明当前环境中的 `flash_attn` 预编译产物与 `torch` 的 C++ ABI 不匹配。本项目不依赖 `flash_attn`，vLLM 会自动回退到 FlashInfer 或内置 FlashAttention，卸载即可：

```bash
pip uninstall -y flash_attn
```

其它已装但未被推理路径使用、且与当前 `torch` ABI 冲突的组件（如 `xformers`、旧版 `mamba_ssm` 等），可按同样方式处理。

### 一次完整运行

以下步骤都在项目根目录下执行。

#### 步骤 0：下载模型权重

模型权重不入库，需要自己下载到 `models/` 目录下。以 Qwen3-30B-A3B 为例：

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B \
    --local-dir ./models/Qwen3-30B-A3B 
```

下载完成后核对目录里包含 `config.json`、`tokenizer.json`、`tokenizer_config.json` 以及若干 `model-*.safetensors` 分片。

#### 步骤 1：生成阶段

每个模型独立进程串行跑，避免同一进程内切换 vLLM 实例造成显存残留。

```bash
python scripts/generate_baseline.py \
    --model-name qwen3-30b-a3b \
    --model-path ./models/Qwen3-30B-A3B \
    --out results/prompting_baselines/qwen3-30b-a3b
```

产物：
- `results/prompting_baselines/qwen3-30b-a3b/qwen3-30b-a3b_raw_outputs.jsonl`
- `results/prompting_baselines/qwen3-30b-a3b/qwen3-30b-a3b_run.yaml`

#### 步骤 2：评估阶段

```bash
python scripts/evaluate_baseline.py \
    --run-dir results/prompting_baselines/qwen3-30b-a3b
```

也支持一次传入多个 `--run-dir` 一起评估。

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

## 实验结果

截至最近一次评估，3 个模型的完成情况如下：

| 模型 | 状态 | Overall Accuracy |
|---|---|---:|
| Qwen3-30B-A3B | 已完成 | **65.3%** (620 / 950) |
| GLM-4.7-Flash | 进行中 | — |
| Nemotron-3-Nano-30B-A3B | 已完成 | **59.6%** (566 / 950) |

### Qwen3-30B-A3B 分类明细

（950 题验证集，`temperature=0 / top_p=1 / max_tokens=32768`，enable_thinking=True）

| category | correct | total | weightage | accuracy | contribution |
|---|---:|---:|---:|---:|---:|
| unit_conversion | 159 | 159 | 16.7% | **100.0%** | 16.7% |
| numeral | 158 | 158 | 16.6% | **100.0%** | 16.6% |
| gravity | 159 | 160 | 16.8% | **99.4%** | 16.7% |
| equation_numeric_deduce | 32 | 60 | 6.3% | 53.3% | 3.4% |
| cipher | 67 | 157 | 16.5% | 42.7% | 7.1% |
| bit_manipulation | 42 | 160 | 16.8% | 26.2% | 4.4% |
| equation_numeric_guess | 1 | 14 | 1.5% | 7.1% | 0.1% |
| cryptarithm_deduce | 2 | 66 | 6.9% | 3.0% | 0.2% |
| cryptarithm_guess | 0 | 16 | 1.7% | 0.0% | 0.0% |
| **TOTAL** | **620** | **950** | 100.0% | **65.3%** | 65.3% |

完整 per-example 结果见 `results/prompting_baselines/qwen3-30b-a3b/qwen3-30b-a3b_validation.csv`；错题按类别分文件保存在 `qwen3-30b-a3b_mistakes/` 目录下。950 题中 33 题（3.5%）因 `finish_reason=length` 被截断，平均生成 12,879 tokens。

### Nemotron-3-Nano-30B-A3B 分类明细

（950 题验证集，`temperature=0 / top_p=1 / max_tokens=32768`，enable_thinking=True）

| category | correct | total | weightage | accuracy | contribution |
|---|---:|---:|---:|---:|---:|
| numeral | 158 | 158 | 16.6% | **100.0%** | 16.6% |
| unit_conversion | 138 | 159 | 16.7% | 86.8% | 14.5% |
| gravity | 119 | 160 | 16.8% | 74.4% | 12.5% |
| cipher | 84 | 157 | 16.5% | 53.5% | 8.8% |
| equation_numeric_deduce | 27 | 60 | 6.3% | 45.0% | 2.8% |
| bit_manipulation | 40 | 160 | 16.8% | 25.0% | 4.2% |
| equation_numeric_guess | 0 | 14 | 1.5% | 0.0% | 0.0% |
| cryptarithm_deduce | 0 | 66 | 6.9% | 0.0% | 0.0% |
| cryptarithm_guess | 0 | 16 | 1.7% | 0.0% | 0.0% |
| **TOTAL** | **566** | **950** | 100.0% | **59.6%** | 59.6% |

**重要 caveat**：Nemotron 有 293/950 题（**30.8%**）`finish_reason=length`，被 32K 生成上限截断（对比 Qwen3 只有 3.5%）。按 category 看：

| category | length-截断率 |
|---|---:|
| cryptarithm_guess | 16/16 (**100%**) |
| cryptarithm_deduce | 64/66 (**97.0%**) |
| equation_numeric_guess | 10/14 (71.4%) |
| bit_manipulation | 103/160 (64.4%) |
| equation_numeric_deduce | 24/60 (40.0%) |
| cipher | 48/157 (30.6%) |
| unit_conversion | 28/159 (17.6%) |

平均生成 13,225 tokens/题（Qwen3 是 12,879），但 Nemotron 有更多题接近或打满 32K 上限。**cryptarithm 系列的 0% 主要是被截断，不能直接归因为"模型能力差"**——如果给 Nemotron 更长的 `max_tokens`（如 64K 或 128K），准确率有望明显提升，是可能的 follow-up 实验方向。

### Qwen3 vs Nemotron 横向对比

| 类别 | Qwen3 | Nemotron |
|---|---:|---:|
| numeral | 100.0% | 100.0% |
| unit_conversion | 100.0% | 86.8% |
| gravity | 99.4% | 74.4% |
| cipher | 42.7% | 53.5% |
| bit_manipulation | 26.2% | 25.0% |
| equation_numeric_deduce | 53.3% | 45.0% |
| equation_numeric_guess | 7.1% | 0.0% |
| cryptarithm_deduce | 3.0% | 0.0% |
| cryptarithm_guess | 0.0% | 0.0% |
| **TOTAL** | **65.3%** | **59.6%** |

完整 per-example 结果见 `results/prompting_baselines/nemotron-3-nano-30b-a3b/nemotron-3-nano-30b-a3b_validation.csv`；错题按类别分文件保存在 `nemotron-3-nano-30b-a3b_mistakes/` 目录下。

## 候选模型选择

本轮零样本 baseline 一共评估 3 个模型：

| 模型 | 结构 | 说明 |
|---|---|---|
| **Nemotron-3-Nano-30B-A3B** | MoE hybrid Mamba-Transformer, 30B / A3B | 官方指定的 base model，是所有实验的锚点。zero-shot 分数用来判断不做 LoRA 的下限。 |
| Qwen3-30B-A3B | MoE Transformer, 30B / A3B | 与 Nemotron 结构最接近的对照：同为 30B 总参 / A3B 激活的 MoE，且 Qwen 系列在 reasoning benchmark 上一贯表现强；用来判断 baseline 差距主要来自架构还是训练数据 / 训练方式。 |
| GLM-4.7-Flash | MoE hybrid attention (MLA + MoE), ~30B / A3B | Z.ai 家族的近邻 MoE 对照，规模接近但训练路线不同；用来判断在同规模下不同厂商在推理任务上的分布差异。 |

三个模型统一使用 32K max_tokens、`temperature=0 / top_p=1` 贪心解码，且都开启 thinking 模式（tokenizer 支持时）。参数完全一致，准确率差距只归因于模型本身。

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
