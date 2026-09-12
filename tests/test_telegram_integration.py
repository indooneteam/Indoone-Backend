from __future__ import annotations

import pytest

from app.capabilities import telegram


@pytest.mark.asyncio
async def test_send_message_requires_approval(monkeypatch):
    monkeypatch.setenv("INDOONE_TELEGRAM_BOT_TOKEN", "123:token")
    with pytest.raises(PermissionError):
        await telegram.send_message("123", "hello", approved=False)


@pytest.mark.asyncio
async def test_send_message_calls_telegram(monkeypatch):
    monkeypatch.setenv("INDOONE_TELEGRAM_BOT_TOKEN", "123:token")

    async def fake_call(method, payload=None, timeout=20.0):
        assert method == "sendMessage"
        assert payload == {"chat_id": "123", "text": "hello"}
        return {"message_id": 42, "chat": {"id": 123}, "text": "hello"}

    monkeypatch.setattr(telegram, "_call", fake_call)
    result = await telegram.send_message("123", "hello", approved=True)
    assert result["operation"] == "send_message"
    assert result["message"]["message_id"] == 42


@pytest.mark.asyncio
async def test_probe_requires_bot_token(monkeypatch):
    monkeypatch.delenv("INDOONE_TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="bot token is not configured"):
        await telegram.probe_telegram()


def test_webhook_secret_validation(monkeypatch):
    monkeypatch.setenv("INDOONE_TELEGRAM_WEBHOOK_SECRET", "secret-value")
    assert telegram.validate_webhook_secret("secret-value") is True
    assert telegram.validate_webhook_secret("wrong") is False
    assert telegram.validate_webhook_secret(None) is False
