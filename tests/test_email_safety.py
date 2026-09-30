
from __future__ import annotations

import base64

import pytest

from app.capabilities.email_safety import (
    analyze_gmail_email_safety,
    extract_gmail_readable_message,
)


def _encoded(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_extract_gmail_readable_message() -> None:
    result = extract_gmail_readable_message(
        {
            "message": {
                "id": "m1",
                "threadId": "t1",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Security Team <security@example.com>"},
                        {"name": "Subject", "value": "Verify account"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _encoded("Your account will close soon.")},
                        }
                    ],
                },
            }
        }
    )
    assert result["message_id"] == "m1"
    assert result["subject"] == "Verify account"
    assert result["body"] == "Your account will close soon."


@pytest.mark.asyncio
async def test_analyze_gmail_email_safety_uses_local_ai(monkeypatch) -> None:
    async def fake_get(user_id: str, message_id: str):
        return {
            "message": {
                "id": message_id,
                "payload": {
                    "headers": [{"name": "Subject", "value": "Verify account"}],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {
                                "data": _encoded(
                                    "Send your password within 15 minutes."
                                )
                            },
                        }
                    ],
                },
            }
        }

    class FakeAI:
        seen = ""

        async def generate(self, message, history=None, document_context=""):
            self.seen = message
            return "Classification: phishing. The message asks for a password and creates urgency. Do not use the link."

    fake_ai = FakeAI()
    monkeypatch.setattr("app.capabilities.email_safety.get_gmail_message", fake_get)
    monkeypatch.setattr("app.capabilities.email_safety.LocalAIService", lambda: fake_ai)

    result = await analyze_gmail_email_safety(
        user_id="user-one",
        message_id="m1",
        user_request="Analyze this email.",
    )
    assert "Classification: phishing" in result
    assert "Verify account" in result
    assert "UNTRUSTED DATA" in fake_ai.seen
