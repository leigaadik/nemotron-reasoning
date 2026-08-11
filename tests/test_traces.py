from src.data.traces import (
    IGNORE_INDEX,
    PROMPT_SUFFIX,
    _encode,
    build_tokenized_trace,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10 + ord(char) % 50 for char in text]


def _row(row_id="one"):
    return {
        "id": row_id,
        "source": "test",
        "category": "unit_conversion",
        "prompt": "Convert this",
        "gold_answer": "12.5",
        "derived_answer": "12.5",
        "cot": "The factor is 2.5.",
        "cot_correct": True,
    }


def test_completion_only_labels_mask_prompt():
    trace = build_tokenized_trace(_row(), FakeTokenizer(), max_length=1024)
    prompt_size = trace["num_prompt_tokens"]
    assert all(label == IGNORE_INDEX for label in trace["labels"][:prompt_size])
    assert trace["labels"][prompt_size:] == trace["input_ids"][prompt_size:]
    assert trace["num_loss_tokens"] > 0


def test_qwen_renderer_matches_chat_template():
    from pathlib import Path

    import pytest

    transformers = pytest.importorskip(
        "transformers", reason="Tokenizer dependencies are unavailable"
    )
    AutoTokenizer = transformers.AutoTokenizer

    model_path = Path(__file__).resolve().parents[1] / "models" / "Qwen3-30B-A3B"
    if not model_path.exists():
        pytest.skip("Qwen3-30B-A3B tokenizer is not available locally")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    user = "Question" + PROMPT_SUFFIX
    assistant = "<think>\nReason\n</think>\n\\boxed{42}"
    actual = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    rendered = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\nReason\n</think>\n\n"
        "\\boxed{42}<|im_end|>\n"
    )
    assert actual == _encode(tokenizer, rendered)
