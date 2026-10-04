from __future__ import annotations

import pytest

import app.integrations.whatsapp.ai as whatsapp_ai


@pytest.mark.asyncio
async def test_whatsapp_identity_is_local(monkeypatch):
    async def fail_shared_ai(*args, **kwargs):
        raise AssertionError("shared AI must not be called for identity questions")

    monkeypatch.setattr(whatsapp_ai, "generate_reply", fail_shared_ai)

    reply = await whatsapp_ai.generate_whatsapp_reply("Who developed Indoone?")

    assert "Indoone" in reply
    assert "Google" not in reply
    assert "Gemini" not in reply


@pytest.mark.asyncio
async def test_whatsapp_identity_handles_roman_kannada(monkeypatch):
    async def fail_shared_ai(*args, **kwargs):
        raise AssertionError("shared AI must not be called for identity questions")

    monkeypatch.setattr(whatsapp_ai, "generate_reply", fail_shared_ai)

    reply = await whatsapp_ai.generate_whatsapp_reply("ninna developer yaaru?")

    assert "Indoone" in reply
    assert "Google" not in reply
    assert "Gemini" not in reply


@pytest.mark.asyncio
async def test_whatsapp_normal_message_uses_shared_ai(monkeypatch):
    calls = []

    async def fake_shared_ai(message, history=None):
        calls.append((message, history))
        return "normal Indoone reply"

    monkeypatch.setattr(whatsapp_ai, "generate_reply", fake_shared_ai)

    reply = await whatsapp_ai.generate_whatsapp_reply("Hello Indoone", history=[("user", "Hi")])

    assert reply == "normal Indoone reply"
    assert calls == [("Hello Indoone", [("user", "Hi")])]
