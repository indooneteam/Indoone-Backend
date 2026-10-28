"""The shared matcher, and the guarantee the routers rely on it for.

The bug this exists to prevent: "hi" came back with the holiday calendar, because the
festival lookup tested whether the message appeared anywhere inside a name - and "hi" is
inside "Maha Shivaratri". The same shape of fault turned "a" into New Year's Day and the
verb "check" into a request for CI status.
"""

from __future__ import annotations

import pytest

from app.ai.execution.text_match import is_pleasantry, mentions, words


def test_words_splits_on_punctuation_and_keeps_kannada() -> None:
    assert words("Hi, there! 1 ralli?") == ["hi", "there", "1", "ralli"]
    assert words("ಹಾಯ್ ಹೇಗಿದ್ದೀರ") == ["ಹಾಯ್", "ಹೇಗಿದ್ದೀರ"]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("maha shivaratri", "hi"),
        ("new year's day", "a"),
        ("good friday", "go"),
        ("ugadi", "di"),
        ("makara sankranti", "ra"),
        ("mahanavami / ayudha pooja", "yu"),
        ("deepavali / naraka chaturdashi", "dee"),  # below the prefix floor
    ],
)
def test_a_fragment_is_not_a_mention(name: str, message: str) -> None:
    assert mentions(name, message) is False


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("deepavali / naraka chaturdashi", "deep"),
        ("deepavali / naraka chaturdashi", "deepavali"),
        ("gandhi jayanti", "gandhi"),
        ("gandhi jayanti", "gandhi jayanti"),
        ("maha shivaratri", "shivaratri"),
        ("makara sankranti", "makara sankranti"),
    ],
)
def test_a_word_or_a_long_prefix_is_a_mention(name: str, message: str) -> None:
    assert mentions(name, message) is True


@pytest.mark.parametrize(
    "message",
    ["hi", "Hi!", "hai", "hello", "hey", "namaskara", "namaste", "ಹಾಯ್", "ok", "thanks",
     "good morning", "how are you", "whats up", "help", "yes", "no", "ಧನ್ಯವಾದ", "ಸರಿ"],
)
def test_greetings_are_pleasantries(message: str) -> None:
    assert is_pleasantry(message) is True


@pytest.mark.parametrize(
    "message",
    ["deepavali", "when is deepavali", "unread emails", "checks", "1 ralli", ""],
)
def test_a_request_is_not_a_pleasantry(message: str) -> None:
    assert is_pleasantry(message) is False


def test_every_router_refuses_a_greeting() -> None:
    """The guarantee, stated once: no fast path answers a pleasantry."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.ai.execution.conversation_digest import conversation_history_answer
    from app.plugins.email.email_tools import email_read_fast_path
    from app.plugins.festivals.calendar import festival_answer
    from app.plugins.github.chat_tools import _github_action

    at = datetime(2026, 10, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    history = [("user", "list my repos"), ("assistant", "1. indoone")]

    for greeting in ("hi", "hello", "namaskara", "ಹಾಯ್", "good morning", "thanks"):
        assert festival_answer(greeting, now=at) is None
        assert _github_action(greeting, history) is None
        assert asyncio.run(email_read_fast_path("u1", "c1", greeting)) is None
        assert asyncio.run(conversation_history_answer("u1", "c1", greeting)) is None
