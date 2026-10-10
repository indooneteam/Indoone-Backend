from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from app.ai.service import generate_reply
from app.capabilities.instagram_messaging import send_text_message as send_instagram_text_message
from app.capabilities.store import (
    claim_channel_inbound_event,
    complete_channel_inbound_event,
    get_single_integration_token_user_id,
)
from app.capabilities.telegram import send_message as send_telegram_message
from app.capabilities.whatsapp import send_text_message as send_whatsapp_text_message
from app.ai.conversation_store import ConversationStore

logger = logging.getLogger("indoone.channel_ai_reply")


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _owner_user_id(name: str) -> str:
    return os.getenv(name, "").strip()


def _conversation_id(channel: str, sender_id: str) -> str:
    digest = hashlib.sha256(f"{channel}:{sender_id}".encode("utf-8")).hexdigest()
    return f"channel:{channel}:{digest}"


async def _generate_with_history(owner_user_id: str, channel: str, sender_id: str, text: str) -> str:
    conversation_id = _conversation_id(channel, sender_id)
    store = ConversationStore()
    try:
        history = store.recent(conversation_id, owner_user_id)
    except (ValueError, PermissionError):
        history = []

    reply = await generate_reply(text, history=history)
    store.append(
        conversation_id,
        [("user", text), ("assistant", reply)],
        user_id=owner_user_id,
    )
    return reply


async def process_whatsapp_message(message: dict[str, Any]) -> bool:
    if not _enabled("INDOONE_WHATSAPP_AI_REPLY_ENABLED"):
        return False

    # WhatsApp already uses server-side credentials. The Firebase UID is optional
    # for conversation ownership; use a dedicated internal scope when it is absent.
    owner_user_id = _owner_user_id("INDOONE_WHATSAPP_AI_OWNER_USER_ID") or "system:whatsapp"

    message_id = str(message.get("id") or "").strip()
    sender_id = str(message.get("from") or "").strip()
    message_type = str(message.get("type") or "").strip().lower()
    text = str(message.get("text") or "").strip()
    if not message_id or not sender_id or message_type != "text" or not text:
        return False

    event_key = f"whatsapp:{message_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    try:
        reply = await _generate_with_history(owner_user_id, "whatsapp", sender_id, text)
        await send_whatsapp_text_message(sender_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        logger.exception("WhatsApp AI reply failed", extra={"message_id": message_id})
        return False

    complete_channel_inbound_event(event_key, success=True)
    logger.info("WhatsApp AI reply sent", extra={"message_id": message_id, "sender_id": sender_id})
    return True


async def process_instagram_messaging_event(
    event: dict[str, Any],
    configured_account_id: str = "",
) -> bool:
    if not _enabled("INDOONE_INSTAGRAM_AI_REPLY_ENABLED"):
        return False

    owner_user_id = (
        _owner_user_id("INDOONE_INSTAGRAM_AI_OWNER_USER_ID")
        or get_single_integration_token_user_id("instagram")
        or ""
    )
    if not owner_user_id:
        logger.error("Instagram AI reply needs exactly one connected account or an explicit owner")
        return False

    sender = event.get("sender")
    recipient = event.get("recipient")
    message = event.get("message")
    if not isinstance(sender, dict) or not isinstance(recipient, dict) or not isinstance(message, dict):
        return False

    sender_id = str(sender.get("id") or "").strip()
    recipient_id = str(recipient.get("id") or "").strip()
    message_id = str(message.get("mid") or "").strip()
    text = str(message.get("text") or "").strip()
    if not sender_id or not recipient_id or sender_id == recipient_id or not message_id or not text:
        return False

    expected_account_id = configured_account_id.strip()
    if expected_account_id and recipient_id != expected_account_id:
        return False

    event_key = f"instagram:{message_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    try:
        reply = await _generate_with_history(owner_user_id, "instagram", sender_id, text)
        await send_instagram_text_message(owner_user_id, sender_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        logger.exception("Instagram AI reply failed", extra={"message_id": message_id})
        return False

    complete_channel_inbound_event(event_key, success=True)
    logger.info("Instagram AI reply sent", extra={"message_id": message_id, "sender_id": sender_id})
    return True


async def process_instagram_webhook(
    payload: dict[str, Any],
) -> int:
    replies_scheduled = 0
    entries = payload.get("entry")
    if not isinstance(entries, list):
        return 0
    configured_account_id = os.getenv("INDOONE_INSTAGRAM_ACCOUNT_ID", "").strip()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        messaging = entry.get("messaging")
        if not isinstance(messaging, list):
            continue
        for event in messaging:
            if not isinstance(event, dict):
                continue
            if await process_instagram_messaging_event(event, configured_account_id):
                replies_scheduled += 1

    return replies_scheduled

async def process_telegram_update(update: dict[str, Any]) -> bool:
    """Generate and send a Telegram reply using the existing bot credentials."""
    toggle = os.getenv("INDOONE_TELEGRAM_AI_REPLY_ENABLED")
    if toggle is not None:
        if not _enabled("INDOONE_TELEGRAM_AI_REPLY_ENABLED"):
            return False
    elif not os.getenv("INDOONE_TELEGRAM_BOT_TOKEN", "").strip():
        return False

    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return False
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return False

    chat_id = str(chat.get("id") or "").strip()
    message_id = str(message.get("message_id") or "").strip()
    text = str(message.get("text") or "").strip()
    sender = message.get("from")
    sender_id = str(sender.get("id") or "").strip() if isinstance(sender, dict) else ""
    update_id = str(update.get("update_id") or "").strip()
    if not chat_id or not message_id or not text:
        return False

    dedupe_id = update_id or f"{chat_id}:{message_id}"
    event_key = f"telegram:{dedupe_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    owner_user_id = "system:telegram"
    try:
        reply = await _generate_with_history(owner_user_id, "telegram", sender_id or chat_id, text)
        await send_telegram_message(chat_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        logger.exception("Telegram AI reply failed", extra={"message_id": message_id})
        return False

    complete_channel_inbound_event(event_key, success=True)
    logger.info("Telegram AI reply sent", extra={"message_id": message_id})
    return True

