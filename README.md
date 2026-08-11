# Nemotron Reasoning

Unified Qwen3-30B-A3B LoRA training and evaluation for the NVIDIA Nemotron
Reasoning Challenge.

## Framework

```text
legacy external CoT CSV ─┐
                         ├─> common tokenized JSONL ─> Transformers Trainer
solver-verified synthetic┘                               + Unsloth LoRA
                                                                  |
                                                                  v
                                                             current_950
```

The two data pipelines share the same Qwen chat renderer, explicit
completion-only labels, collator, sampler, optimizer, scheduler, and LoRA
configuration. Inference uses one vLLM runner and scoring implementation for
the current_950 benchmark suite.

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

## Run Evaluations

Run each adapter separately with `scripts/infer.py`, then score the raw outputs with `scripts/score.py`:

```bash
python scripts/infer.py \
  --run-name legacy-cot-transformers \
  --base-model ./models/Qwen3-30B-A3B \
  --adapter outputs/adapters/legacy-cot-transformers \
  --suite configs/eval/current_950.yaml \
  --out results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950 \
  --force

python scripts/score.py \
  --suite configs/eval/current_950.yaml \
  --raw-outputs results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950/legacy-cot-transformers__current_950_raw_outputs.jsonl \
  --out results/adapter_eval/qwen3-30b/legacy-cot-transformers/current_950
```

Repeat with `synthetic-cot-transformers` and `low-quality-cot-transformers` by changing the run name, adapter path, and output directory.

## Documentation

- `docs/reproduction.md`: end-to-end experiment reproduction steps.
- `docs/experiments/synthetic_qwen_pilot.md`: completed synthetic pilot results.
- `docs/experiments/`: historical experiment records.

## Main Files

```text
configs/data/                    legacy and synthetic data configurations
configs/training/                common training config plus two overrides
configs/eval/                    current_950 benchmark suite
data/                             source CSVs and training splits
docs/                             reproduction guide and experiment records
notebooks/evaluation/             reference notebooks (adapter validation, metric)
scripts/build_data.py            unified data builder
scripts/train.py                 unified Transformers Trainer entry
scripts/infer.py                 unified base/LoRA vLLM entry
scripts/score.py                 unified scoring entry
scripts/run_lora.sh              template: single LoRA training run
src/data/                        data adapters, schema rendering, validation
src/generators/                  problem generators (nemotron, 7 categories)
src/training/                    Trainer and collator
src/inference/                   vLLM runner
src/evaluation/                  suites, scoring, reporting
src/prompting/                   zero-shot prompting-baseline dataset loader
src/providers/                   inference provider backends (local vLLM)
tests/                            unit tests for data/config/validation modules
```
