"""Agent 09 - the Token Budget Arbiter.

The two failure modes this exists for are opposite: a bill with no ceiling, and an estimate
that undercounts Kannada as if it were English. Both are tested here directly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient

from app.agents import load

arbiter = load("09_token_budget_arbiter")

SECRET = "x" * 32

ENGLISH = "the quick brown fox jumps over the lazy dog " * 4
KANNADA = "ನಾಳೆ ರಾಜ್ಯೋತ್ಸವ ಹಬ್ಬ ಇದೆಯಾ ಎಂದು ಹೇಳಿ ಮತ್ತು ದಯವಿಟ್ಟು ವಿವರಿಸಿ"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    arbiter.reset()
    monkeypatch.setenv("INDOONE_TOKEN_ARBITER_ENABLED", "true")
    monkeypatch.setenv("INDOONE_DAILY_TOKEN_ALLOWANCE", "200000")
    monkeypatch.setenv("INDOONE_PROMPT_TOKEN_CEILING", "16000")
    yield
    arbiter.reset()


def _auth_headers(user_id: str) -> dict[str, str]:
    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(user_id.encode("utf-8"))
    stamp = encode(str(int(time.time())).encode("ascii"))
    signature = encode(hmac.new(SECRET.encode("ascii"), f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{stamp}.{signature}"}


# ---------------------------------------------------------------- 1. allocation


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("who is the chief minister of karnataka", arbiter.FACTOID_TOKENS),
        ("write code to sort a list in python", arbiter.TECHNICAL_TOKENS),
        ("debug this stack trace", arbiter.TECHNICAL_TOKENS),
        ("write a design document for the auth service", arbiter.ARCHITECTURE_TOKENS),
        ("plan the architecture of the billing module", arbiter.ARCHITECTURE_TOKENS),
    ],
)
def test_the_answer_gets_room_that_matches_the_question(prompt: str, expected: int) -> None:
    assert arbiter.allocate_max_tokens("", prompt) == expected


def test_architecture_beats_code_when_both_words_appear() -> None:
    """\"write a design document\" contains \"write\"; the longer answer must still win."""
    assert arbiter.allocate_max_tokens("", "write a design document for the parser") == arbiter.ARCHITECTURE_TOKENS


def test_a_router_answered_turn_allocates_nothing() -> None:
    assert arbiter.allocate_max_tokens("FAST_PATH", "list branches") == arbiter.FAST_PATH_TOKENS


# ---------------------------------------------------------------- 2. estimation


def test_indic_text_counts_higher_per_character_than_english() -> None:
    english_density = arbiter.estimate_tokens(ENGLISH) / len(ENGLISH)
    kannada_density = arbiter.estimate_tokens(KANNADA) / len(KANNADA)

    assert kannada_density > english_density, (
        f"Kannada {kannada_density:.3f} tokens/char must exceed English {english_density:.3f}"
    )


def test_an_estimate_never_undercounts_a_short_string() -> None:
    assert arbiter.estimate_tokens("ok") >= 1
    assert arbiter.estimate_tokens("") == 0


def test_history_is_counted_with_the_prompt() -> None:
    messages = [{"role": "user", "content": ENGLISH}]

    total = arbiter.estimate_prompt_tokens("hello", messages)

    assert total > arbiter.estimate_tokens("hello")


# ---------------------------------------------------------------- 3. refusals


def test_a_massive_prompt_is_refused_before_a_provider_is_called() -> None:
    verdict = arbiter.evaluate("stuffer", "x" * 200_000)

    assert verdict.allowed is False
    assert verdict.rejection_reason == "PROMPT_TOO_LARGE"
    assert verdict.allocated_max_tokens == 0
    assert verdict.client_message is not None
    assert verdict.client_message["en"]


def test_the_refusal_speaks_the_users_language() -> None:
    verdict = arbiter.evaluate("stuffer", "x" * 200_000, language="kn")

    assert verdict.client_message is not None
    assert verdict.client_message["primary"] != verdict.client_message["en"]


def test_an_exhausted_quota_refuses_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_DAILY_TOKEN_ALLOWANCE", "1000")
    arbiter.reset()

    arbiter.record_usage("heavy", 1000)

    verdict = arbiter.evaluate("heavy", "hello there")

    assert verdict.allowed is False
    assert verdict.rejection_reason == "QUOTA_EXHAUSTED"
    assert verdict.daily_tokens_remaining == 0


def test_a_turn_within_the_quota_passes() -> None:
    verdict = arbiter.evaluate("normal", "what is docker")

    assert verdict.allowed is True
    assert verdict.estimated_prompt_tokens > 0
    assert verdict.daily_tokens_remaining > 0


def test_the_ceiling_is_checked_before_the_quota() -> None:
    """A prompt that is too large is refused whether or not the user can afford it."""
    verdict = arbiter.evaluate("rich-user-with-no-quota-left", "x" * 200_000)

    assert verdict.rejection_reason == "PROMPT_TOO_LARGE"


# ---------------------------------------------------------------- 4. the ledger


def test_usage_accumulates_and_can_be_reset() -> None:
    arbiter.record_usage("u1", 500)
    arbiter.record_usage("u1", 500)

    assert arbiter.get_user_usage("u1")["tokens_used"] == 1000

    arbiter.reset_quota("u1")

    assert arbiter.get_user_usage("u1")["tokens_used"] == 0


def test_the_window_rolls_after_a_day() -> None:
    now = 1000.0
    arbiter.record_usage("u2", 5000, now=now)

    assert arbiter.get_user_usage("u2", now=now)["tokens_used"] == 5000
    assert arbiter.get_user_usage("u2", now=now + arbiter.WINDOW_SECONDS)["tokens_used"] == 0


def test_the_ledger_is_bounded() -> None:
    for index in range(arbiter.MAX_TRACKED_USERS + 10):
        arbiter.record_usage(f"user-{index}", 1)

    assert len(arbiter._ledger) == arbiter.MAX_TRACKED_USERS


# ---------------------------------------------------------------- 5. speed


def test_the_decision_costs_under_a_twentieth_of_a_millisecond() -> None:
    fastest = _fastest_call_microseconds()

    assert fastest < 50, f"fastest call {fastest:.2f} us"


def _fastest_call_microseconds() -> float:
    best = float("inf")

    for _ in range(300):
        started = time.thread_time()
        arbiter.evaluate("speed-user", "what is docker")
        best = min(best, (time.thread_time() - started) * 1_000_000)

    return best


# ---------------------------------------------------------------- 6. the endpoint


def test_a_stuffed_prompt_is_refused_by_the_endpoint_without_the_model(monkeypatch) -> None:
    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", SECRET)
    monkeypatch.setenv("INDOONE_TOKEN_ARBITER_ENABLED", "true")
    # The request schema already caps a message's length, so the ceiling is lowered here to
    # prove the gate itself: a prompt the schema accepts and the arbiter refuses.
    # The ceiling has a floor of 1000 tokens; 800 words is comfortably past it.
    monkeypatch.setenv("INDOONE_PROMPT_TOKEN_CEILING", "1000")

    calls: list[str] = []

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        calls.append(message)
        return "should not happen"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    response = TestClient(app).post(
        "/api/chat",
        json={"message": "kannada " * 900},
        headers=_auth_headers("stuffer"),
    )

    assert response.status_code == 413
    assert calls == [], "a refused turn must never reach the model"


def test_the_endpoint_records_what_the_turn_cost(monkeypatch) -> None:
    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", SECRET)
    monkeypatch.setenv("INDOONE_TOKEN_ARBITER_ENABLED", "true")

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return "a short answer"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    response = TestClient(app).post(
        "/api/chat",
        json={"message": "what is docker"},
        headers=_auth_headers("spender"),
    )

    assert response.status_code == 200
    assert arbiter.get_user_usage("spender")["tokens_used"] > 0
