from __future__ import annotations

import logging
import os
from typing import Any

from app.capabilities.channel_common import _enabled, _generate_with_history
from app.capabilities.store import claim_channel_inbound_event, complete_channel_inbound_event
from app.telegram.service import send_message as send_telegram_message

logger = logging.getLogger("indoone.telegram.ai_reply")


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
