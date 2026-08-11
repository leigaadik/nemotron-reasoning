from pathlib import Path

from src.data.legacy import clean_external_cot, iter_legacy_cot


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_clean_external_cot_removes_old_boxed_answers():
    assert clean_external_cot("reasoning\\boxed{old}\n") == "reasoning"


def test_clean_external_cot_removes_existing_assistant_tail():
    value = "reasoning\\boxed{draft}\n</think>\n\\boxed{final}"
    cleaned = clean_external_cot(value)
    assert cleaned == "reasoning"
    assert "</think>" not in cleaned


def test_legacy_adapter_emits_common_pre_token_schema():
    row = next(iter_legacy_cot(REPO_ROOT / "data/train_split_with_cot.csv"))
    assert row["source"] == "legacy_external_cot"
    assert row["cot_quality"] == "external_unverified"
    assert row["cot_correct"] is None
    assert row["gold_answer"] == row["target_answer"]
    assert row["cot"]
