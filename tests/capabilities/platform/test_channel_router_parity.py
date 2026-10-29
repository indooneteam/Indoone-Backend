"""A channel answers the same question the same way as the app.

The WhatsApp and Telegram path had the calendar, the research fast path and the tool loop -
but not the conversation digest and not the mailbox read. The same question therefore got a
different answer on a channel than in the app, and on a channel there is no second screen to
correct it from.
"""

from __future__ import annotations

import asyncio

from app.ai.memory.conversation_store import ConversationStore
from app.capabilities.platform import channel_common


def _wire(tmp_path, monkeypatch, *, model_reply: str | None = None):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(conversation_db)
    )
    # the digest keeps its own reference to the store, so it needs the same one
    from app.ai.execution import conversation_digest

    monkeypatch.setattr(
        conversation_digest, "ConversationStore", lambda: ConversationStore(conversation_db)
    )

    calls: list[str] = []

    async def fake_generate_reply(message, history=None, document_context="", **kwargs):
        calls.append(message)
        return model_reply if model_reply is not None else "model answer"

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    return calls, ConversationStore(conversation_db)


def test_a_thread_question_is_answered_by_the_digest_on_a_channel(tmp_path, monkeypatch) -> None:
    calls, store = _wire(tmp_path, monkeypatch)
    store.append(
        channel_common._conversation_id("whatsapp", "9199"),
        [("user", "deepavali planning"), ("assistant", "Deepavali is on 8 November 2026.")],
        user_id="u1",
    )

    reply = asyncio.run(
        channel_common._generate_with_history("u1", "whatsapp", "9199", "summarise this chat")
    )

    assert reply is not None
    assert "deepavali" in reply.casefold() or "Deepavali" in reply
    assert calls == [], "the digest should answer without asking the model"


def test_a_promise_on_a_channel_is_replaced(tmp_path, monkeypatch) -> None:
    calls, _store = _wire(tmp_path, monkeypatch, model_reply="One moment, I will fetch that.")

    async def fake_research(*args, **kwargs):
        return "The listing date has not been announced."

    monkeypatch.setattr(channel_common, "answer_instead_of_promise", fake_research)

    reply = asyncio.run(
        channel_common._generate_with_history("u1", "whatsapp", "9198", "jio ipo estakke listing agide")
    )

    assert reply == "The listing date has not been announced."
