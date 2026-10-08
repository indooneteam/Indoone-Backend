import pytest
import torch

from app.ai.inference import LocalModelRuntime
from app.ai.local_engine import LocalAIEngine
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


def test_select_next_token_uses_greedy_choice_at_zero_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])
    assert LocalModelRuntime._select_next_token(logits, 0.0) == 1


def test_select_next_token_rejects_negative_temperature() -> None:
    logits = torch.tensor([[0.1, 2.5, 1.0]])

    with pytest.raises(ValueError, match="non-negative"):
        LocalModelRuntime._select_next_token(logits, -0.1)


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


def test_preformatted_prompt_with_trailing_newline_is_not_wrapped_twice(tmp_path) -> None:
    tokenizer = BPETokenizer.train(
        "<instruction>\nSay hello\n</instruction>\n<response>\nHello\n</response>\n",
        vocab_size=64,
        min_frequency=1,
    )
    runtime = object.__new__(LocalModelRuntime)
    runtime.tokenizer = tokenizer
    prompt_ids = runtime._prompt_ids(
        "<instruction>\nSay hello\n</instruction>\n<response>\n"
    )
    decoded = tokenizer.decode(prompt_ids)
    assert decoded.count("<instruction>") == 1
    assert decoded.count("<response>") == 1


def test_runtime_has_no_question_specific_answer_retrieval() -> None:
    assert not hasattr(LocalModelRuntime, "_retrieved_response")
    assert not hasattr(LocalModelRuntime, "_supplied_evidence_response")
    assert not hasattr(LocalModelRuntime, "_extract_user_request")


def test_cached_forward_matches_full_forward() -> None:
    torch.manual_seed(7)
    model = IndooneTransformer(
        vocab_size=48,
        block_size=32,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    ).eval()
    tokens = torch.randint(0, 48, (1, 6))

    full_logits, _ = model(tokens)
    cached_outputs = []
    cached_logits, cache = model.forward_cached(tokens[:, :3])
    cached_outputs.append(cached_logits)
    for index in range(3, tokens.size(1)):
        cached_logits, cache = model.forward_cached(tokens[:, index:index + 1], cache)
        cached_outputs.append(cached_logits)
    combined_cached_logits = torch.cat(cached_outputs, dim=1)

    assert combined_cached_logits.shape == full_logits.shape
    assert torch.allclose(combined_cached_logits, full_logits, atol=1e-5, rtol=1e-4)


def test_cached_forward_rejects_multi_token_decode_steps() -> None:
    model = IndooneTransformer(
        vocab_size=16,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    ).eval()
    _, cache = model.forward_cached(torch.tensor([[1, 2, 3]]))

    with pytest.raises(ValueError, match="one generated token"):
        model.forward_cached(torch.tensor([[4, 5]]), cache)


def test_cached_forward_respects_block_size() -> None:
    model = IndooneTransformer(
        vocab_size=16,
        block_size=4,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    ).eval()
    _, cache = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))

    with pytest.raises(ValueError, match="exceeds model block size"):
        model.forward_cached(torch.tensor([[5]]), cache)


def test_prompt_fitting_preserves_current_request_and_reserves_output_tokens() -> None:
    tokenizer = BPETokenizer.train(
        "<instruction>\ncurrent request\n</instruction>\n<response>\n"
        "CONVERSATION HISTORY: old message new message context",
        vocab_size=128,
        min_frequency=1,
    )
    runtime = object.__new__(LocalModelRuntime)
    runtime.tokenizer = tokenizer
    runtime.model = type("ModelConfig", (), {"block_size": 96})()

    prompt = (
        "current request"
        "\n\nCONVERSATION HISTORY:"
        "\nuser: " + ("old context " * 80)
        + "\nassistant: latest context"
    )

    fitted = runtime._fit_prompt_to_context(prompt, max_new_tokens=48)

    assert len(fitted) <= 48
    decoded = tokenizer.decode(fitted)
    assert "current request" in decoded


def test_language_constraint_keeps_english_tokens(tmp_path) -> None:
    tokenizer = BPETokenizer.train(
        "Hello world. ಕನ್ನಡ ನಮಸ್ಕಾರ.",
        vocab_size=128,
        min_frequency=1,
    )
    runtime = object.__new__(LocalModelRuntime)
    runtime.tokenizer = tokenizer
    runtime._language_token_ids = {}

    logits = torch.zeros((1, tokenizer.vocab_size))
    constrained = runtime._apply_language_constraint(logits, "English")

    assert torch.isfinite(constrained).any()
    assert torch.all(constrained <= 0)
