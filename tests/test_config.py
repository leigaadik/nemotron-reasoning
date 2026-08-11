from src.config import deep_merge, load_config


def test_deep_merge_preserves_shared_training_fields():
    merged = deep_merge(
        {"training": {"learning_rate": 2e-4, "epochs": 1}},
        {"training": {"epochs": 2}},
    )
    assert merged == {"training": {"learning_rate": 2e-4, "epochs": 2}}


def test_training_overrides_extend_common_base():
    configs = {
        name: load_config(f"configs/training/{name}.yaml")[0]
        for name in ("legacy", "synthetic", "low_quality")
    }
    baseline = configs["legacy"]
    baseline_training = {
        k: v for k, v in baseline["training"].items() if k != "output_dir"
    }

    for name, config in configs.items():
        training = {k: v for k, v in config["training"].items() if k != "output_dir"}
        assert training == baseline_training, name
        assert config["lora"] == baseline["lora"], name
        assert config["model"] == baseline["model"], name

    train_jsonls = {config["paths"]["train_jsonl"] for config in configs.values()}
    adapter_dirs = {config["paths"]["adapter_dir"] for config in configs.values()}
    output_dirs = {config["training"]["output_dir"] for config in configs.values()}
    assert len(train_jsonls) == len(configs)
    assert len(adapter_dirs) == len(configs)
    assert len(output_dirs) == len(configs)
