"""The digest answers the way people ask.

"what did we discuss" was handled; "summarise this chat" and "what did we talk about" were
not, so they reached a model that cannot see the thread - the same question, a different
answer depending on the wording.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.execution.conversation_digest import conversation_history_answer
from app.ai.memory.conversation_store import ConversationStore


@pytest.fixture()
def thread(tmp_path, monkeypatch) -> str:
    store = ConversationStore(tmp_path / "c.sqlite3")
    store.append(
        "c-digest",
        [("user", "deepavali planning maadona"), ("assistant", "Deepavali is on 8 November 2026.")],
        user_id="u1",
    )
    return "c-digest"


@pytest.mark.parametrize(
    "message",
    [
        "what did we discuss",
        "what did we talk about",
        "summarise this chat",
        "summarize the conversation",
        "summary of this thread",
        "namma maatu helu",
        "conversation summary",
    ],
)
def test_the_question_is_recognised(message: str, thread: str) -> None:
    assert asyncio.run(conversation_history_answer("u1", thread, message)) is not None


@pytest.mark.parametrize(
    "message",
    ["hi", "thanks", "when is deepavali", "unread emails", "list branches", "web research maadu gold rate"],
)
def test_another_question_is_left_alone(message: str, thread: str) -> None:
    assert asyncio.run(conversation_history_answer("u1", thread, message)) is None
