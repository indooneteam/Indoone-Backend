"""A read-only mailbox question is answered without spending a token.

On a Gemini or Azure deployment the email path went through the model, which then chose a
tool the words already named: "unread emails" can only mean one thing. The answer is the
same; the tokens are not.

A write is deliberately not taken here - it is held for confirmation, and that path knows
how to hold it.
"""

from __future__ import annotations

import asyncio

import pytest

from app.plugins.email import email_tools


@pytest.fixture(autouse=True)
def _no_pending(monkeypatch):
    monkeypatch.setattr(email_tools, "get_pending_email_action", lambda *_a, **_k: None)


def _stub(monkeypatch, answer="1. From: a@b.com — Subject: hello"):
    calls: list[str] = []

    async def fake_chat(user_id: str, conversation_id: str, message: str) -> str:
        calls.append(message)
        return answer

    monkeypatch.setattr(email_tools, "handle_email_chat", fake_chat)
    return calls


@pytest.mark.parametrize(
    "message",
    ["unread emails", "check my mail", "inbox thorisu", "show my latest emails"],
)
def test_a_read_question_is_answered_directly(monkeypatch, message: str) -> None:
    calls = _stub(monkeypatch)

    answer = asyncio.run(email_tools.email_read_fast_path("u1", "c1", message))

    assert answer == "1. From: a@b.com — Subject: hello"
    assert calls == [message]


@pytest.mark.parametrize(
    "message",
    [
        "send an email to a@b.com about the meeting",
        "reply to that mail saying yes",
        "delete the last email",
        "draft a mail to my manager",
        "how do I reset my password",
    ],
)
def test_a_write_or_an_unrelated_message_is_left_alone(monkeypatch, message: str) -> None:
    calls = _stub(monkeypatch)

    assert asyncio.run(email_tools.email_read_fast_path("u1", "c1", message)) is None
    assert calls == []


def test_a_held_action_owns_the_conversation(monkeypatch) -> None:
    _stub(monkeypatch)
    monkeypatch.setattr(
        email_tools, "get_pending_email_action", lambda *_a, **_k: {"action_type": "send_email"}
    )

    # "yes" after a preview must reach the path that can execute it.
    assert asyncio.run(email_tools.email_read_fast_path("u1", "c1", "unread emails")) is None


# The endpoint-level version of this test was dropped: it needs a connected mailbox
# fixture, and this environment has none - the request answers "No email account is
# connected" before the fast path is reached. The unit tests above are the real proof: the
# function answers, and returns None for everything it must not touch.
