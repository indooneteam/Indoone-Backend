"""A greeting is not a request.

"hi" came back with the holiday calendar, because the festival lookup tested whether the
query appeared *anywhere inside* a name - and "hi" is inside "Maha Shivaratri". "a", "go",
"di", "ra" and "yu" did the same against other names. The GitHub router had the same shape
of fault from the other side: the bare verb "check" was read as a request for CI status, so
"check the text" listed a repository's check runs.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.plugins.festivals.calendar import festival_answer
from app.plugins.github import chat_tools

AT = datetime(2026, 10, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
HISTORY = [("user", "list my repos"), ("assistant", "1. indoone\nhttps://github.com/indooneteam/indoone")]


@pytest.mark.parametrize(
    "message",
    [
        "hi", "Hi", "hai", "hello", "hey", "hii", "hlo",
        "namaskara", "namaste", "ಹಾಯ್", "ಹಾಯ್ ಹೇಗಿದ್ದೀರ",
        "ok", "thanks", "good morning", "how are you", "whats up",
        "a", "go", "di", "ra", "yu", "ma", "ka", "an", "ha", "ti",
    ],
)
def test_a_greeting_or_a_fragment_is_not_a_festival_question(message: str) -> None:
    assert festival_answer(message, now=AT) is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("deep", "Deepavali"),
        ("deepavali", "Deepavali"),
        ("when is deepavali", "Deepavali"),
        ("gandhi", "Gandhi Jayanti"),
        ("gandhi jayanti", "Gandhi Jayanti"),
        ("ugadi", "Ugadi"),
        ("when is ugadi", "Ugadi"),
        ("shivaratri", "Maha Shivaratri"),
        ("makara sankranti", "Makara Sankranti"),
        ("deepavali yavaga", "Deepavali"),
    ],
)
def test_the_real_questions_still_answer(message: str, expected: str) -> None:
    answer = festival_answer(message, now=AT)

    assert answer is not None
    assert expected in answer


@pytest.mark.parametrize("message", ["check", "check the text", "check my mail", "check this code"])
def test_the_verb_check_is_not_a_ci_request(message: str) -> None:
    assert chat_tools._github_action(message, HISTORY) is None


@pytest.mark.parametrize("message", ["checks", "show checks", "ci status", "checks for 1"])
def test_the_noun_checks_still_routes(message: str) -> None:
    action = chat_tools._github_action(message, HISTORY)

    assert action is not None
    assert isinstance(action, tuple)
    assert action[0] == "github_list_check_runs"
