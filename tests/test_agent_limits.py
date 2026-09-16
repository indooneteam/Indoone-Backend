from __future__ import annotations

from app.ai.agent import MAX_AGENT_APPROVAL_TOKENS, _bounded_approval_tokens


def test_approval_token_materialization_is_bounded() -> None:
    consumed = 0

    def tokens():
        nonlocal consumed
        for index in range(MAX_AGENT_APPROVAL_TOKENS + 100):
            consumed += 1
            yield f"token-{index}"

    result = _bounded_approval_tokens(tokens())

    assert len(result) == MAX_AGENT_APPROVAL_TOKENS
    assert consumed == MAX_AGENT_APPROVAL_TOKENS


def test_approval_token_materialization_accepts_short_iterables() -> None:
    assert _bounded_approval_tokens(["a", "b"]) == ("a", "b")
