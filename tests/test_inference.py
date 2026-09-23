import torch

from app.ai.inference import LocalModelRuntime
from app.ai.local_engine import LocalAIEngine
from app.ai.tokenizer import BPETokenizer


def test_select_next_token_uses_greedy_choice_at_zero_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    assert LocalModelRuntime._select_next_token(logits, 0.0) == 1


def test_select_next_token_rejects_negative_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        LocalModelRuntime._select_next_token(logits, -0.1)



def test_repetition_penalty_ignores_prompt_tokens() -> None:
    logits = torch.tensor([[1.0, -2.0, 3.0]])
    adjusted = LocalModelRuntime._apply_repetition_penalty(
        logits,
        [],
        1.08,
    )
    assert torch.equal(adjusted, logits)


def test_ngram_blocking_ignores_prompt_tokens() -> None:
    logits = torch.zeros((1, 8))
    adjusted = LocalModelRuntime._block_repeated_ngram(
        logits,
        [1, 2, 3],
        3,
    )
    assert torch.equal(adjusted, logits)
def test_local_engine_select_next_token_uses_greedy_choice_at_zero_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    assert LocalAIEngine._select_next_token(logits, 0.0) == 1


def test_clean_completion_stops_before_malformed_instruction_marker() -> None:
    cleaned = LocalModelRuntime._clean_completion(
        "A useful answer.<instructioninstruction>Next training example"
    )
    assert cleaned == "A useful answer."


def test_preformatted_prompt_is_not_wrapped_twice(tmp_path) -> None:
    tokenizer = BPETokenizer.train(
        "<instruction>\nSay hello\n</instruction>\n<response>\nHello\n</response>\n",
        vocab_size=64,
        min_frequency=1,
    )
    runtime = object.__new__(LocalModelRuntime)
    runtime.tokenizer = tokenizer
    prompt_ids = runtime._prompt_ids(
        "<instruction>\nSay hello\n</instruction>\n<response>"
    )
    assert tokenizer.decode(prompt_ids).count("<instruction>") == 1
