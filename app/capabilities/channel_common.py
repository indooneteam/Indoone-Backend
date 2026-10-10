from __future__ import annotations

import hashlib
import os

from app.ai.service import generate_reply
from app.ai.conversation_store import ConversationStore

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
