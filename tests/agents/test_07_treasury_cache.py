"""Agent 07 - the Treasury.

A cache is a promise that two questions are the same question. These tests are mostly about
the three ways that promise is broken: private data, answers that move, and actions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient

from app.agents import treasury as _treasury

cache = _treasury()

SECRET = "x" * 32


@pytest.fixture(autouse=True)
def _fresh():
    cache.reset()
    yield
    cache.reset()


def _auth_headers(user_id: str) -> dict[str, str]:
    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(user_id.encode("utf-8"))
    stamp = encode(str(int(time.time())).encode("ascii"))
    signature = encode(hmac.new(SECRET.encode("ascii"), f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{stamp}.{signature}"}


# ---------------------------------------------------------------- 1. hit and miss


def test_a_second_identical_turn_is_a_hit() -> None:
    assert cache.get("what is docker", "en", "CASUAL").cache_hit is False

    assert cache.put("what is docker", "en", "CASUAL", response="Docker is a container runtime.") is True

    hit = cache.get("what is docker", "en", "CASUAL")

    assert hit.cache_hit is True
    assert hit.cached_response == "Docker is a container runtime."


def test_a_hit_costs_under_the_probe_budget() -> None:
    cache.put("what is docker", "en", "CASUAL", response="Docker is a container runtime.")

    started = time.process_time()
    for _ in range(1000):
        cache.get("what is docker", "en", "CASUAL")
    mean_ms = (time.process_time() - started) * 1000 / 1000

    assert mean_ms < cache.PROBE_BUDGET_MILLISECONDS, f"{mean_ms:.4f} ms"


# ---------------------------------------------------------------- 2. canonical collapse


def test_spelling_variants_reach_one_entry() -> None:
    first = cache.canonical_key("naale raja na?", "kn", "CASUAL")
    second = cache.canonical_key("nale raja na", "kn", "CASUAL")

    assert first == second


def test_language_and_register_are_part_of_the_key() -> None:
    kannada = cache.canonical_key("when is deepavali", "kn", "CASUAL")
    english = cache.canonical_key("when is deepavali", "en", "CASUAL")
    polite = cache.canonical_key("when is deepavali", "en", "HONORIFIC")

    assert len({kannada, english, polite}) == 3


def test_an_unknown_word_is_never_folded() -> None:
    """Folding is a known vocabulary, not a guess."""
    assert "kubernetes" in cache.canonical_key("what is kubernetes", "en", "CASUAL")


# ---------------------------------------------------------------- 3. the three refusals


@pytest.mark.parametrize(
    ("query", "subject"),
    [
        ("read email", "email"),
        ("git diff", "code"),
        ("what did we discuss", "conversation"),
        ("unread emails", "email"),
    ],
)
def test_private_and_live_subjects_are_never_stored(query: str, subject: str) -> None:
    assert cache.should_store(query, subject) is False
    assert cache.put(query, "en", "CASUAL", response="secret", subject=subject) is False
    assert cache.get(query, "en", "CASUAL").cache_hit is False


@pytest.mark.parametrize(
    "query",
    ["naale raja na?", "today festival", "this month holidays", "ಇಂದು ಹಬ್ಬ ಇದೆಯಾ"],
)
def test_a_question_whose_answer_moves_is_never_stored(query: str) -> None:
    """The date is right today and wrong tomorrow, so the answer is not a fact to keep."""
    assert cache.should_store(query) is False


@pytest.mark.parametrize(
    "query",
    ["send the mail", "merge the pr", "delete this thread", "create a branch"],
)
def test_an_action_is_never_stored(query: str) -> None:
    assert cache.should_store(query) is False


@pytest.mark.parametrize(
    "query",
    ["what is docker", "how do transactions work in postgres", "when is deepavali"],
)
def test_a_public_fact_is_stored(query: str) -> None:
    assert cache.should_store(query) is True


# ---------------------------------------------------------------- 4. capacity and TTL


def test_the_cache_never_exceeds_its_capacity() -> None:
    for index in range(cache.MAX_ENTRIES + 1):
        cache.put(f"question number {index}", "en", "CASUAL", response=f"answer {index}")

    assert cache.stats()["entries"] == cache.MAX_ENTRIES


def test_the_least_recently_used_entry_goes_first() -> None:
    for index in range(cache.MAX_ENTRIES):
        cache.put(f"question {index}", "en", "CASUAL", response=f"answer {index}")

    # touch the oldest so it is no longer the least recently used
    assert cache.get("question 0", "en", "CASUAL").cache_hit is True
    cache.put("one more question", "en", "CASUAL", response="answer")

    assert cache.get("question 0", "en", "CASUAL").cache_hit is True
    assert cache.get("question 1", "en", "CASUAL").cache_hit is False


def test_an_expired_entry_is_a_miss() -> None:
    cache.put("what is docker", "en", "CASUAL", response="Docker is a container runtime.", ttl_seconds=10)

    assert cache.get("what is docker", "en", "CASUAL", now=time.monotonic() + 11).cache_hit is False


def test_the_calendar_keeps_an_answer_for_a_day_and_documentation_for_six_hours() -> None:
    assert cache.ttl_for("when is deepavali") == cache.HOLIDAY_TTL_SECONDS
    assert cache.ttl_for("what is docker") == cache.GENERAL_TTL_SECONDS


def test_an_entry_can_be_invalidated() -> None:
    cache.put("what is docker", "en", "CASUAL", response="wrong answer")
    key = cache.canonical_key("what is docker", "en", "CASUAL")

    assert cache.invalidate(key) is True
    assert cache.get("what is docker", "en", "CASUAL").cache_hit is False


# ---------------------------------------------------------------- 5. the endpoint


def test_a_cached_turn_costs_no_provider_call(monkeypatch) -> None:
    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", SECRET)
    monkeypatch.setenv("INDOONE_TREASURY_CACHE_ENABLED", "true")

    calls: list[str] = []

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        calls.append(message)
        return "Docker is a container runtime."

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    client = TestClient(app)
    body = {"message": "what is docker"}
    headers = _auth_headers("treasury-user")

    first = client.post("/api/chat", json=body, headers=headers)
    assert first.status_code == 200
    assert len(calls) == 1, "the first turn must reach the model"

    second = client.post("/api/chat", json=body, headers=headers)

    assert second.status_code == 200
    assert len(calls) == 1, "the second turn must be served from the treasury"
    assert second.json()["reply"] == first.json()["reply"]
