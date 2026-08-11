from pathlib import Path

import pytest

from src.evaluation.suites import load_suite


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config", "expected"),
    [("configs/eval/current_950.yaml", 950), ("configs/eval/reference_8224.yaml", 8224)],
)
def test_suite_size_and_unique_ids(config, expected):
    suite = load_suite(REPO_ROOT / config)
    assert len(suite.examples) == expected
    assert len({example.id for example in suite.examples}) == expected
