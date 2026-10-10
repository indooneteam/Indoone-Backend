import asyncio

from app.ai.conversation_store import ConversationStore
from app.capabilities import channel_ai_reply, store


def test_channel_event_claim_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))

    assert store.claim_channel_inbound_event("whatsapp:message-1") is True
    assert store.claim_channel_inbound_event("whatsapp:message-1") is False

    store.complete_channel_inbound_event("whatsapp:message-1", success=False)
    assert store.claim_channel_inbound_event("whatsapp:message-1", retry_after_seconds=0) is True


def test_whatsapp_incoming_message_replies_with_indoon_ai(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "indoone-user-1")

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_ai_reply,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str]] = []

    async def fake_generate_reply(message: str, history=None, document_context: str = "") -> str:
        assert message == "Hello Indoone"
        assert history == []
        return "Hello! How can I help you?"

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False) -> dict[str, object]:
        sent.append((to, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_ai_reply, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(channel_ai_reply, "send_whatsapp_text_message", fake_send)

    message = {
        "id": "wamid.test-1",
        "from": "919999999999",
        "type": "text",
        "text": "Hello Indoone",
    }

    assert asyncio.run(channel_ai_reply.process_whatsapp_message(message)) is True
    assert asyncio.run(channel_ai_reply.process_whatsapp_message(message)) is False
    assert sent == [("919999999999", "Hello! How can I help you?")]


def test_instagram_message_replies_with_indoon_ai(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_OWNER_USER_ID", "indoone-user-1")
    monkeypatch.setenv("INDOONE_INSTAGRAM_ACCOUNT_ID", "ig-business-1")

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_ai_reply,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str, str]] = []

    async def fake_generate_reply(message: str, history=None, document_context: str = "") -> str:
        assert message == "Hi from Instagram"
        assert history == []
        return "Hi! Thanks for messaging Indoone."

    async def fake_send(user_id: str, recipient_id: str, text: str, approved: bool = False) -> dict[str, object]:
        sent.append((user_id, recipient_id, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_ai_reply, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(channel_ai_reply, "send_instagram_text_message", fake_send)

    event = {
        "sender": {"id": "ig-customer-1"},
        "recipient": {"id": "ig-business-1"},
        "timestamp": 123,
        "message": {"mid": "mid.test-1", "text": "Hi from Instagram"},
    }

    assert asyncio.run(
        channel_ai_reply.process_instagram_messaging_event(event, "ig-business-1")
    ) is True
    assert asyncio.run(
        channel_ai_reply.process_instagram_messaging_event(event, "ig-business-1")
    ) is False
    assert sent == [
        (
            "indoone-user-1",
            "ig-customer-1",
            "Hi! Thanks for messaging Indoone.",
        )
    ]


def test_whatsapp_reply_works_without_owner_uid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.delenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", raising=False)

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_ai_reply,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str]] = []

    async def fake_generate_reply(message: str, history=None, document_context: str = "") -> str:
        assert message == "Hello without UID"
        return "Reply without UID"

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False) -> dict[str, object]:
        sent.append((to, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_ai_reply, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(channel_ai_reply, "send_whatsapp_text_message", fake_send)

    message = {
        "id": "wamid.test-no-owner-uid",
        "from": "919999999999",
        "type": "text",
        "text": "Hello without UID",
    }

    assert asyncio.run(channel_ai_reply.process_whatsapp_message(message)) is True
    assert sent == [("919999999999", "Reply without UID")]
    history = ConversationStore(conversation_db).recent(
        channel_ai_reply._conversation_id("whatsapp", "919999999999"),
        "system:whatsapp",
    )
    assert history == [("user", "Hello without UID"), ("assistant", "Reply without UID")]
