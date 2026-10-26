import asyncio

from app.ai.memory.conversation_store import ConversationStore
from app.capabilities.connectors import store
from app.capabilities.platform import channel_common
from app.channels.instagram import ai_reply as instagram_ai_reply
from app.channels.telegram import ai_reply as telegram_ai_reply
from app.channels.whatsapp import ai_reply as whatsapp_ai_reply


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
        channel_common,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str]] = []

    async def fake_generate_reply(
        message: str, history=None, document_context: str = "", **kwargs
    ) -> str:
        assert message == "Hello Indoone"
        assert history == []
        return "Hello! How can I help you?"

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False) -> dict[str, object]:
        sent.append((to, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    message = {
        "id": "wamid.test-1",
        "from": "919999999999",
        "type": "text",
        "text": "Hello Indoone",
    }

    assert asyncio.run(whatsapp_ai_reply.process_whatsapp_message(message)) is True
    assert asyncio.run(whatsapp_ai_reply.process_whatsapp_message(message)) is False
    assert sent == [("919999999999", "Hello! How can I help you?")]


def test_instagram_message_replies_with_indoon_ai(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_OWNER_USER_ID", "indoone-user-1")
    monkeypatch.setenv("INDOONE_INSTAGRAM_ACCOUNT_ID", "ig-business-1")

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_common,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str, str]] = []

    async def fake_generate_reply(
        message: str, history=None, document_context: str = "", **kwargs
    ) -> str:
        assert message == "Hi from Instagram"
        assert history == []
        return "Hi! Thanks for messaging Indoone."

    async def fake_send(user_id: str, recipient_id: str, text: str, approved: bool = False) -> dict[str, object]:
        sent.append((user_id, recipient_id, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(instagram_ai_reply, "send_instagram_text_message", fake_send)

    event = {
        "sender": {"id": "ig-customer-1"},
        "recipient": {"id": "ig-business-1"},
        "timestamp": 123,
        "message": {"mid": "mid.test-1", "text": "Hi from Instagram"},
    }

    assert asyncio.run(
        instagram_ai_reply.process_instagram_messaging_event(event, "ig-business-1")
    ) is True
    assert asyncio.run(
        instagram_ai_reply.process_instagram_messaging_event(event, "ig-business-1")
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
        channel_common,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )

    sent: list[tuple[str, str]] = []

    async def fake_generate_reply(
        message: str, history=None, document_context: str = "", **kwargs
    ) -> str:
        assert message == "Hello without UID"
        return "Reply without UID"

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False) -> dict[str, object]:
        sent.append((to, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    message = {
        "id": "wamid.test-no-owner-uid",
        "from": "919999999999",
        "type": "text",
        "text": "Hello without UID",
    }

    assert asyncio.run(whatsapp_ai_reply.process_whatsapp_message(message)) is True
    assert sent == [("919999999999", "Reply without UID")]
    history = ConversationStore(conversation_db).recent(
        channel_common._conversation_id("whatsapp", "919999999999"),
        "system:whatsapp",
    )
    assert history == [("user", "Hello without UID"), ("assistant", "Reply without UID")]

def test_instagram_reply_resolves_existing_connected_account_without_uid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_REPLY_ENABLED", "true")
    monkeypatch.delenv("INDOONE_INSTAGRAM_AI_OWNER_USER_ID", raising=False)
    monkeypatch.delenv("INDOONE_INSTAGRAM_ACCOUNT_ID", raising=False)
    store.upsert_integration_token(
        "instagram-account-owner",
        "instagram",
        b"encrypted-token-placeholder",
        None,
        "Bearer",
        "instagram_business_manage_messages",
        None,
    )

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_common,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )
    sent: list[tuple[str, str, str]] = []

    async def fake_generate_reply(
        message: str, history=None, document_context: str = "", **kwargs
    ) -> str:
        assert message == "Hello from Instagram"
        return "Instagram reply"

    async def fake_send(user_id: str, recipient_id: str, text: str, approved: bool = False) -> dict[str, object]:
        sent.append((user_id, recipient_id, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(instagram_ai_reply, "send_instagram_text_message", fake_send)

    event = {
        "sender": {"id": "ig-customer-7"},
        "recipient": {"id": "ig-business-2"},
        "message": {"mid": "mid.no-owner-uid", "text": "Hello from Instagram"},
    }
    assert asyncio.run(instagram_ai_reply.process_instagram_messaging_event(event)) is True
    assert sent == [("instagram-account-owner", "ig-customer-7", "Instagram reply")]


def test_telegram_reply_uses_existing_bot_token_without_owner_uid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_TELEGRAM_BOT_TOKEN", "configured-bot-token")
    monkeypatch.delenv("INDOONE_TELEGRAM_AI_REPLY_ENABLED", raising=False)
    monkeypatch.delenv("INDOONE_TELEGRAM_AI_OWNER_USER_ID", raising=False)

    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setattr(
        channel_common,
        "ConversationStore",
        lambda: ConversationStore(conversation_db),
    )
    sent: list[tuple[str, str]] = []

    async def fake_generate_reply(
        message: str, history=None, document_context: str = "", **kwargs
    ) -> str:
        assert message == "Hello from Telegram"
        return "Telegram reply"

    async def fake_send(chat_id: str, text: str, approved: bool = False, parse_mode: str = "") -> dict[str, object]:
        sent.append((chat_id, text))
        assert approved is True
        return {"sent": True}

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(telegram_ai_reply, "send_telegram_message", fake_send)

    update = {
        "update_id": 9876,
        "message": {
            "message_id": 25,
            "chat": {"id": 123456},
            "from": {"id": 123456},
            "text": "Hello from Telegram",
        },
    }
    assert asyncio.run(telegram_ai_reply.process_telegram_update(update)) is True
    assert asyncio.run(telegram_ai_reply.process_telegram_update(update)) is False
    assert sent == [("123456", "Telegram reply")]



# ---- the channels answer through the same loop as the app ------------------ #
# These handlers used to call the model directly, so a message on WhatsApp was
# answered with no tools at all, no date, and no statement about connected accounts.
# The app and the channels were two different assistants sharing a model.


def _message(text: str = "Hello Indoone") -> dict:
    """A webhook message after parsing: the text is already flattened to a string."""

    return {
        "id": "wamid.parity-1",
        "from": "919999999999",
        "type": "text",
        "text": text,
    }


def test_a_channel_message_goes_through_the_tool_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "cap.db"))
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "owner-1")
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(tmp_path / "conv.sqlite3")
    )
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")

    calls: list[dict] = []

    async def fake_loop(user_id, conversation_id, message, **kwargs):
        calls.append({"user_id": user_id, "message": message, **kwargs})
        return "Looped answer"

    async def fake_generate_reply(*_args, **_kwargs):
        raise AssertionError("the plain reply must not be used when the loop answers")

    sent: list[tuple[str, str]] = []

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False):
        sent.append((to, text))
        return {"sent": True}

    monkeypatch.setattr(channel_common, "run_tool_loop", fake_loop)
    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    assert asyncio.run(whatsapp_ai_reply.process_whatsapp_message(_message())) is True

    assert calls and calls[0]["message"] == "Hello Indoone"
    assert sent == [("919999999999", "Looped answer")]


def test_the_channel_instruction_carries_the_date_and_the_accounts(tmp_path, monkeypatch) -> None:
    """A model with no date answers a date question from training memory."""

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "cap.db"))
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "owner-1")
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(tmp_path / "conv.sqlite3")
    )

    seen: dict = {}

    async def fake_loop(user_id, conversation_id, message, **kwargs):
        seen.update(kwargs)
        return "ok"

    async def fake_send(*_args, **_kwargs):
        return {"sent": True}

    monkeypatch.setattr(channel_common, "run_tool_loop", fake_loop)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    asyncio.run(whatsapp_ai_reply.process_whatsapp_message(_message()))

    instruction = str(seen.get("extra_system_instruction") or "")
    assert "Today is" in instruction
    assert "IST" in instruction


def test_the_loop_returning_nothing_falls_back_to_a_plain_reply(tmp_path, monkeypatch) -> None:
    """The loop returns None when the model chose no tool - the ordinary case."""

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "cap.db"))
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "owner-1")
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(tmp_path / "conv.sqlite3")
    )

    async def fake_loop(*_args, **_kwargs):
        return None

    async def fake_generate_reply(message, history=None, **kwargs):
        return "Plain answer"

    sent: list[tuple[str, str]] = []

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False):
        sent.append((to, text))
        return {"sent": True}

    monkeypatch.setattr(channel_common, "run_tool_loop", fake_loop)
    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    asyncio.run(whatsapp_ai_reply.process_whatsapp_message(_message()))

    assert sent == [("919999999999", "Plain answer")]


def test_a_loop_failure_still_answers_the_user(tmp_path, monkeypatch) -> None:
    """A provider hiccup must not cost an answer on a channel nobody can retry from."""

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "cap.db"))
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "owner-1")
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(tmp_path / "conv.sqlite3")
    )

    async def exploding_loop(*_args, **_kwargs):
        raise RuntimeError("provider down")

    async def fake_generate_reply(message, history=None, **kwargs):
        return "Plain answer"

    sent: list[tuple[str, str]] = []

    async def fake_send(to: str, text: str, approved: bool = False, preview_url: bool = False):
        sent.append((to, text))
        return {"sent": True}

    monkeypatch.setattr(channel_common, "run_tool_loop", exploding_loop)
    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    asyncio.run(whatsapp_ai_reply.process_whatsapp_message(_message()))

    assert sent == [("919999999999", "Plain answer")]


def test_an_old_refusal_is_dropped_from_the_channel_history(tmp_path, monkeypatch) -> None:
    """The model reads its own last answer as evidence, so a stale "no access" poisons it."""

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "cap.db"))
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_OWNER_USER_ID", "owner-1")
    monkeypatch.setattr(
        channel_common, "ConversationStore", lambda: ConversationStore(tmp_path / "conv.sqlite3")
    )
    monkeypatch.delenv("INDOONE_MODEL_BACKEND", raising=False)

    seen: dict = {}

    async def fake_generate_reply(message, history=None, **kwargs):
        seen["history"] = history
        return "ok"

    async def fake_send(*_args, **_kwargs):
        return {"sent": True}

    store = ConversationStore()
    conversation_id = channel_common._conversation_id("whatsapp", "919999999999")
    store.append(
        conversation_id,
        [
            ("user", "how many branches in indoone"),
            ("assistant", "I don't have access to your GitHub account."),
        ],
        user_id="owner-1",
    )

    monkeypatch.setattr(channel_common, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(whatsapp_ai_reply, "send_whatsapp_text_message", fake_send)

    asyncio.run(whatsapp_ai_reply.process_whatsapp_message(_message("try again")))

    history = seen.get("history") or []
    assert all("don't have access" not in content for _role, content in history)
