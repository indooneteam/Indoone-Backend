from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from app.ai.service import generate_reply
from app.capabilities.control_center import record_control_event, replies_enabled
from app.capabilities.telegram import send_message as send_telegram_message
from app.capabilities.instagram_messaging import send_text_message as send_instagram_text_message
from app.capabilities.store import (
    claim_channel_inbound_event,
    complete_channel_inbound_event,
)
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


def _record_reply_outcome(channel: str, event_id: str, status: str, path: str, http_status: int | None = None) -> None:
    """Record reply results without persisting provider event IDs or message content."""
    digest = hashlib.sha256(f"{channel}:{event_id}:{status}".encode("utf-8")).hexdigest()
    try:
        record_control_event(
            channel=channel,
            event_type="reply",
            status=status,
            path=path,
            http_status=http_status,
            dedupe_key=f"reply:{channel}:{digest}",
        )
    except Exception:
        logger.exception("Control Center reply metric recording failed", extra={"channel": channel, "status": status})


async def process_whatsapp_message(message: dict[str, Any]) -> bool:
    message_id = str(message.get("id") or "").strip()
    sender_id = str(message.get("from") or "").strip()
    message_type = str(message.get("type") or "").strip().lower()
    text = str(message.get("text") or "").strip()
    if not message_id or not sender_id:
        return False

    route = "/api/integrations/whatsapp/webhook"
    if not replies_enabled("whatsapp"):
        _record_reply_outcome("whatsapp", message_id, "skipped", route, 200)
        return False
    if message_type != "text" or not text:
        _record_reply_outcome("whatsapp", message_id, "skipped", route, 200)
        return False

    owner_user_id = _owner_user_id("INDOONE_WHATSAPP_AI_OWNER_USER_ID")
    if not owner_user_id:
        _record_reply_outcome("whatsapp", message_id, "failed", route, 503)
        logger.error("WhatsApp AI reply is enabled but owner user id is not configured")
        return False

    event_key = f"whatsapp:{message_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    try:
        reply = await _generate_with_history(owner_user_id, "whatsapp", sender_id, text)
        await send_whatsapp_text_message(sender_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        _record_reply_outcome("whatsapp", message_id, "failed", route, 502)
        logger.exception("WhatsApp AI reply failed", extra={"message_id_hash": hashlib.sha256(message_id.encode("utf-8")).hexdigest()})
        return False

    complete_channel_inbound_event(event_key, success=True)
    _record_reply_outcome("whatsapp", message_id, "success", route, 200)
    logger.info("WhatsApp AI reply sent")
    return True


async def process_instagram_messaging_event(
    event: dict[str, Any],
    configured_account_id: str = "",
) -> bool:
    sender = event.get("sender")
    recipient = event.get("recipient")
    message = event.get("message")
    if not isinstance(sender, dict) or not isinstance(recipient, dict) or not isinstance(message, dict):
        return False

    sender_id = str(sender.get("id") or "").strip()
    recipient_id = str(recipient.get("id") or "").strip()
    message_id = str(message.get("mid") or "").strip()
    text = str(message.get("text") or "").strip()
    if not sender_id or not recipient_id or sender_id == recipient_id or not message_id:
        return False

    expected_account_id = configured_account_id.strip()
    if expected_account_id and recipient_id != expected_account_id:
        return False

    route = "/api/integrations/instagram/webhook"
    if not replies_enabled("instagram"):
        _record_reply_outcome("instagram", message_id, "skipped", route, 200)
        return False
    if not text:
        _record_reply_outcome("instagram", message_id, "skipped", route, 200)
        return False

    owner_user_id = _owner_user_id("INDOONE_INSTAGRAM_AI_OWNER_USER_ID")
    if not owner_user_id:
        _record_reply_outcome("instagram", message_id, "failed", route, 503)
        logger.error("Instagram AI reply is enabled but owner user id is not configured")
        return False

    event_key = f"instagram:{message_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    try:
        reply = await _generate_with_history(owner_user_id, "instagram", sender_id, text)
        await send_instagram_text_message(owner_user_id, sender_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        _record_reply_outcome("instagram", message_id, "failed", route, 502)
        logger.exception("Instagram AI reply failed", extra={"message_id_hash": hashlib.sha256(message_id.encode("utf-8")).hexdigest()})
        return False

    complete_channel_inbound_event(event_key, success=True)
    _record_reply_outcome("instagram", message_id, "success", route, 200)
    logger.info("Instagram AI reply sent")
    return True


async def process_telegram_update(update: dict[str, Any]) -> bool:
    """Generate and send a Telegram reply for one authenticated incoming bot update."""
    update_id = str(update.get("update_id") or "").strip()
    message = update.get("message")
    route = "/api/telegram/webhook/incoming"
    if not update_id or not isinstance(message, dict):
        if update_id:
            _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False

    chat = message.get("chat")
    sender = message.get("from")
    if not isinstance(chat, dict):
        _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False
    chat_id = str(chat.get("id") or "").strip()
    sender_id = str(sender.get("id") or "").strip() if isinstance(sender, dict) else chat_id
    text = str(message.get("text") or "").strip()
    if isinstance(sender, dict) and sender.get("is_bot") is True:
        _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False
    if not chat_id:
        _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False

    if not replies_enabled("telegram"):
        _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False
    if not text:
        _record_reply_outcome("telegram", update_id, "skipped", route, 200)
        return False

    owner_user_id = _owner_user_id("INDOONE_TELEGRAM_AI_OWNER_USER_ID")
    if not owner_user_id:
        _record_reply_outcome("telegram", update_id, "failed", route, 503)
        logger.error("Telegram AI reply is enabled but owner user id is not configured")
        return False

    event_key = f"telegram:{update_id}"
    if not claim_channel_inbound_event(event_key):
        return False

    try:
        reply = await _generate_with_history(owner_user_id, "telegram", sender_id or chat_id, text)
        await send_telegram_message(chat_id, reply, approved=True)
    except Exception:
        complete_channel_inbound_event(event_key, success=False)
        _record_reply_outcome("telegram", update_id, "failed", route, 502)
        logger.exception("Telegram AI reply failed", extra={"update_id_hash": hashlib.sha256(update_id.encode("utf-8")).hexdigest()})
        return False

    complete_channel_inbound_event(event_key, success=True)
    _record_reply_outcome("telegram", update_id, "success", route, 200)
    logger.info("Telegram AI reply sent")
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
