import torch

from app.ai.inference import LocalModelRuntime


def test_select_next_token_uses_greedy_choice_at_zero_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    assert LocalModelRuntime._select_next_token(logits, 0.0) == 1


def test_select_next_token_rejects_negative_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        LocalModelRuntime._select_next_token(logits, -0.1)
