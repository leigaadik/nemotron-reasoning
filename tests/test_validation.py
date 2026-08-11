from src.data.validation import validate_tokenized_rows


def _row(row_id="synthetic-unit-test"):
    return {
        "id": row_id,
        "category": "unit_conversion",
        "prompt": "not an evaluation prompt",
        "input_ids": [1, 2, 3, 4],
        "labels": [-100, -100, 3, 4],
        "num_prompt_tokens": 2,
    }


def test_common_schema_validation_passes_clean_row():
    report = validate_tokenized_rows([_row()], max_length=8, overlap_policy="forbid")
    assert report["error_count"] == 0
    assert report["rows"] == 1


def test_common_schema_validation_rejects_duplicate_id():
    report = validate_tokenized_rows(
        [_row(), _row()], max_length=8, overlap_policy="allow"
    )
    assert report["error_count"] == 1
