from __future__ import annotations

import logging
import os
from typing import Any

from app.capabilities.channel_common import _enabled, _generate_with_history, _owner_user_id
from app.capabilities.store import (
    claim_channel_inbound_event,
    complete_channel_inbound_event,
    get_single_integration_token_user_id,
)
from app.instagram.capabilities.instagram_messaging import send_text_message as send_instagram_text_message

logger = logging.getLogger("indoone.instagram.ai_reply")


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
