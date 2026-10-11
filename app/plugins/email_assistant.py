from __future__ import annotations

import re
from app.capabilities.store import get_integration_token_metadata
from app.plugins.gmail.service import list_gmail_messages
from app.plugins.zoho_mail.service import list_accounts, list_messages

EMAIL_KEYWORDS = ("email", "mail", "gmail", "zoho", "inbox", "finance", "draft", "delete", "send", "ಮೇಲ್", "ಇಮೇಲ್")


async def handle_email_chat(user_id: str, conversation_id: str, message: str) -> str | None:
    text = message.strip()
    if not any(word in text.casefold() for word in EMAIL_KEYWORDS):
        return None
    return "Email connection tool routing is not yet complete."
