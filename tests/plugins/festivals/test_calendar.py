"""Festival answers come from the file, cost nothing, and never invent a date.

The dates cannot be computed - they follow a lunar calendar and a state gazette - so the
only honest source is a stored list with its source recorded. These tests hold both halves:
the lookup answers, and an ordinary message is left to the model.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.plugins.festivals.calendar import festival_answer

INDIA = ZoneInfo("Asia/Kolkata")


def at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 9, 0, tzinfo=INDIA)


def test_today_is_answered_from_the_file() -> None:
    answer = festival_answer("today festival", now=at(2026, 1, 14))

    assert answer is not None
    assert "Makara Sankranti" in answer
    assert "2026-01-14" in answer


def test_a_kannada_question_gets_a_kannada_answer() -> None:
    answer = festival_answer("ಇಂದು ಯಾವ ಹಬ್ಬ", now=at(2026, 3, 19))

    assert answer is not None
    assert "ಯುಗಾದಿ" in answer
    assert "ಸರ್ಕಾರಿ ರಜೆ: ಹೌದು" in answer


def test_tomorrow_looks_one_day_ahead() -> None:
    answer = festival_answer("naale yava habba", now=at(2026, 3, 18))

    assert answer is not None
    assert "Ugadi" in answer
    assert "2026-03-19" in answer


def test_a_named_festival_returns_its_date() -> None:
    answer = festival_answer("when is Deepavali", now=at(2026, 6, 1))

    assert answer is not None
    # Both the central-list date and the Karnataka state date are in the file.
    assert "2026-11-08" in answer
    assert "2026-11-09" in answer


def test_a_month_lists_its_holidays() -> None:
    answer = festival_answer("this month holidays", now=at(2026, 10, 1))

    assert answer is not None
    assert "Vijayadasami" in answer
    assert "Gandhi Jayanti" in answer


def test_a_day_with_nothing_on_it_says_so() -> None:
    answer = festival_answer("today festival", now=at(2026, 7, 7))

    assert answer == "No festival or holiday today."


def test_an_ordinary_message_is_left_to_the_model() -> None:
    # No calendar words, no festival name: the router must not answer this.
    assert festival_answer("how do I reset my password", now=at(2026, 3, 19)) is None
    assert festival_answer("", now=at(2026, 3, 19)) is None


def test_the_chat_endpoint_answers_without_calling_the_model(monkeypatch) -> None:
    """The proof that no tokens are spent: the model is made to raise."""
    import base64
    import hashlib
    import hmac
    import os
    import time

    from fastapi.testclient import TestClient

    from app.api.chat import chat as chat_module
    from app.main import app

    os.environ["INDOONE_AUTH_SECRET"] = "x" * 32

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    user = encode(b"user-one")
    stamp = encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{stamp}".encode("ascii")
    signature = encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    headers = {"Authorization": f"Bearer {user}.{stamp}.{signature}"}

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("a festival question must not reach the model")

    monkeypatch.setattr(chat_module, "generate_reply", must_not_run)
    monkeypatch.setattr("app.api.chat.chat._admit", lambda *_a, **_k: None)

    response = TestClient(app).post(
        "/api/chat", json={"message": "today festival"}, headers=headers
    )

    assert response.status_code == 200
    assert response.json()["reply"]
