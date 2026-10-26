"""The model sees four messages; the thread is still stored, and asking for it is free.

Every turn used to carry twelve messages of history. Most of it was noise, and it was paid
for on every request. The window is now four - and a question about earlier in the
conversation is answered from a digest instead of a model call.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.execution import conversation_digest
from app.ai.execution.history_window import DEFAULT_HISTORY_LIMIT, history_window

THREAD = [
    ("user", "list my repos"),
    ("assistant", "1. indoone\nhttps://github.com/indooneteam/indoone"),
    ("user", "create a branch feat/x in 1"),
    ("assistant", "Created the branch feat/x. Reply CONFIRM to go ahead."),
    ("user", "now show the commits"),
    ("assistant", "Recent commits: a1b2c3 — fix: routing"),
    ("user", "thanks"),
    ("assistant", "You're welcome."),
    ("user", "check the deploys"),
    ("assistant", "The last deploy succeeded."),
    ("user", "and the branches?"),
    ("assistant", "1. main\n2. develop\n3. terminal\n4. web"),
    ("user", "ok"),
    ("assistant", "Noted."),
    ("user", "one more thing"),
    ("assistant", "Go ahead."),
]


def test_the_model_window_is_four_messages() -> None:
    assert DEFAULT_HISTORY_LIMIT == 4
    assert len(history_window(THREAD)) == 4
    assert history_window(THREAD) == THREAD[-4:]


def test_a_short_thread_is_passed_through() -> None:
    short = THREAD[:2]

    assert history_window(short) == short
    assert history_window(None) == []


def test_the_window_keeps_the_most_recent_exchange() -> None:
    window = history_window(THREAD)

    # The ordinal the user just typed must still have its list in view.
    assert window[-1] == THREAD[-1]


def test_a_digest_lists_topics_and_decisions() -> None:
    text = conversation_digest.digest(THREAD)

    assert "Topics:" in text
    assert "Decisions:" in text
    # A decision-shaped answer is picked up, a plain one is not required.
    assert "Created the branch feat/x" in text
    # The whole transcript is never returned.
    assert len(text) < 1200


def test_a_kannada_question_gets_a_kannada_digest() -> None:
    text = conversation_digest.digest(THREAD, kannada=True)

    assert "ವಿಷಯಗಳು:" in text
    assert "ನಿರ್ಧಾರಗಳು:" in text


def test_an_empty_thread_says_so() -> None:
    assert "Nothing is stored" in conversation_digest.digest([])


@pytest.mark.parametrize(
    "message",
    [
        "namma conversation check maadu",
        "what did we discuss earlier",
        "chat history thorisu",
        "ಸಂಭಾಷಣೆ ತೋರಿಸು",
    ],
)
def test_an_explicit_question_triggers_the_digest(monkeypatch, message: str) -> None:
    class FakeStore:
        def recent(self, conversation_id, user_id=None):
            return THREAD

    monkeypatch.setattr(conversation_digest, "ConversationStore", FakeStore)

    answer = asyncio.run(
        conversation_digest.conversation_history_answer("u1", "c1", message)
    )

    assert answer is not None
    assert "Topics:" in answer or "ವಿಷಯಗಳು:" in answer


@pytest.mark.parametrize(
    "message",
    ["unread emails", "create a branch", "what is the weather", "hello"],
)
def test_an_ordinary_message_does_not_trigger_it(monkeypatch, message: str) -> None:
    class FakeStore:
        def recent(self, conversation_id, user_id=None):
            raise AssertionError("the store must not be read for an ordinary message")

    monkeypatch.setattr(conversation_digest, "ConversationStore", FakeStore)

    assert (
        asyncio.run(conversation_digest.conversation_history_answer("u1", "c1", message))
        is None
    )
