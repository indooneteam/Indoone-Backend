"""One question, one router.

The fast paths run in order, so the first one that claims a message wins. Measured before
this: "what did we discuss about deepavali" and "unread emails about deepavali" were both
answered with Deepavali's date - the calendar claimed a message about the thread and a
message about the mailbox, because both of them mentioned a festival.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.ai.execution.text_match import subject_of
from app.plugins.festivals.calendar import festival_answer

AT = datetime(2026, 10, 29, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


@pytest.mark.parametrize(
    ("message", "subject"),
    [
        ("unread emails about deepavali", "email"),
        ("gmail alli deepavali mail ide", "email"),
        ("what did we discuss about deepavali", "conversation"),
        ("namma maatu helu deepavali bagge", "conversation"),
        ("summarise this chat about ugadi", "conversation"),
        ("check the branch for deepavali", "code"),
        ("deepavali commit ide", "code"),
        ("when is deepavali", None),
        ("today festival", None),
        ("this month holidays", None),
    ],
)
def test_the_subject_is_read_from_the_message(message: str, subject: str | None) -> None:
    assert subject_of(message) == subject


@pytest.mark.parametrize(
    "message",
    [
        "what did we discuss about deepavali",
        "namma maatu helu deepavali bagge",
        "unread emails about deepavali",
        "check the branch for deepavali",
    ],
)
def test_the_calendar_stands_down_for_another_subject(message: str) -> None:
    assert festival_answer(message, now=AT) is None


@pytest.mark.parametrize(
    "message",
    ["when is deepavali", "deepavali", "today festival", "this month holidays", "gandhi jayanti"],
)
def test_the_calendar_still_answers_its_own_subject(message: str) -> None:
    assert festival_answer(message, now=AT) is not None
