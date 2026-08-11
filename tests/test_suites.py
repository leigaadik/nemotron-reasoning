from pathlib import Path


from src.evaluation.suites import load_suite


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_950_size_and_unique_ids():
    suite = load_suite(REPO_ROOT / "configs/eval/current_950.yaml")
    assert len(suite.examples) == 950
    assert len({example.id for example in suite.examples}) == 950
