from __future__ import annotations

import re

_EMAIL_TERMS = re.compile(r"\b(email|emails|mail|mails|inbox|gmail|zoho|draft|unread|finance|send|compose|delete|read|search|ಹೊಸ ಮೇಲ್|ಇತ್ತೀಚಿನ ಮೇಲ್)\b", re.IGNORECASE)


async def handle_email_chat(user_id: str, conversation_id: str, message: str) -> str | None:
    """Handle mail-specific requests here; return None for ordinary chat."""
    if not user_id.strip() or not conversation_id.strip() or not message.strip():
        return None
    if not _EMAIL_TERMS.search(message):
        return None
    return "Email assistant router is ready."
