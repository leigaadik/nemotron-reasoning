# Nemotron Reasoning

Unified Qwen3-30B-A3B LoRA training and evaluation for the NVIDIA Nemotron
Reasoning Challenge.

## Framework

```text
legacy external CoT CSV ─┐
                         ├─> common tokenized JSONL ─> Transformers Trainer
solver-verified synthetic┘                               + Unsloth LoRA
                                                                  |
                                        ┌─────────────────────────┴──────────┐
                                        v                                    v
                                   current_950                         reference_8224
```

The two data pipelines share the same Qwen chat renderer, explicit
completion-only labels, collator, sampler, optimizer, scheduler, and LoRA
configuration. Inference uses one vLLM runner and scoring implementation for
both benchmark suites.

## Environment

```bash
conda create -n nemotron python=3.10 -y
conda activate nemotron
pip install -r requirements.txt
```

Local model weights are expected at:

```text
models/Qwen3-30B-A3B/
```

## Build Training Data

Historical external CoT:

```bash
python scripts/build_data.py --config configs/data/legacy.yaml
```

Solver-verified synthetic CoT:

```bash
python scripts/build_data.py --config configs/data/synthetic_pilot.yaml
```

Both outputs use the same pre-tokenized JSONL schema. Legacy benchmark overlap
is measured and recorded; synthetic benchmark overlap is forbidden.

## Train an Adapter

Copy and edit the template config, then run:
```bash
cp configs/training/legacy.yaml configs/training/<experiment>.yaml
# edit: experiment.name, paths.train_jsonl, paths.adapter_dir, training.output_dir

python scripts/train.py --config configs/training/<experiment>.yaml
# or: bash scripts/run_lora.sh
```
Adapters are written to:

```text
outputs/adapters/legacy-cot-transformers/
outputs/adapters/synthetic-cot-transformers/
```

## Run Four Evaluations

```bash
bash scripts/run_evaluation_matrix.sh \
  ./models/Qwen3-30B-A3B \
  outputs/adapters/legacy-cot-transformers \
  outputs/adapters/synthetic-cot-transformers \
  results/evaluation_matrix/qwen3-30b
```

This runs four independent model processes:

| Adapter | current_950 | reference_8224 |
|---|---|---|
| legacy-cot-transformers | evaluation | evaluation |
| synthetic-cot-transformers | evaluation | evaluation |

The combined result is written to `matrix_summary.csv`. Each matrix cell also
contains raw outputs, run metadata, per-example results, category summaries,
mistake files, and logs.

## Documentation

- `docs/unified_framework.md`: framework and command reference.
- `docs/experiments/synthetic_qwen_pilot.md`: completed synthetic pilot results.
- `docs/experiments/`: historical experiment records.

## Main Files

```text
configs/data/                    legacy and synthetic data configurations
configs/training/                common training config plus two overrides
configs/eval/                    two benchmark suites
data/                             source CSVs (train split, reference eval set)
docs/                             framework reference and experiment records
notebooks/evaluation/             reference notebooks (adapter validation, metric)
scripts/build_data.py            unified data builder
scripts/train.py                 unified Transformers Trainer entry
scripts/infer.py                 unified base/LoRA vLLM entry
scripts/score.py                 unified scoring entry
scripts/run_lora.sh              template: single LoRA training run
scripts/run_evaluation_matrix.sh run and summarize the four evaluations
src/data/                        data adapters, schema rendering, validation
src/generators/                  problem generators (nemotron, 7 categories)
src/training/                    Trainer and collator
src/inference/                   vLLM runner
src/evaluation/                  suites, scoring, reporting
src/prompting/                   zero-shot prompting-baseline dataset loader
src/providers/                   inference provider backends (local vLLM)
tests/                            unit tests for data/config/validation modules
```
