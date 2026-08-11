from src.config import deep_merge, load_config


def test_deep_merge_preserves_shared_training_fields():
    merged = deep_merge(
        {"training": {"learning_rate": 2e-4, "epochs": 1}},
        {"training": {"epochs": 2}},
    )
    assert merged == {"training": {"learning_rate": 2e-4, "epochs": 2}}


def test_training_overrides_extend_common_base():
    legacy, _ = load_config("configs/training/legacy.yaml")
    synthetic, _ = load_config("configs/training/synthetic.yaml")
    legacy_training = {k: v for k, v in legacy["training"].items() if k != "output_dir"}
    synthetic_training = {
        k: v for k, v in synthetic["training"].items() if k != "output_dir"
    }
    assert legacy_training == synthetic_training
    assert legacy["lora"] == synthetic["lora"]
    assert legacy["paths"]["train_jsonl"] != synthetic["paths"]["train_jsonl"]
