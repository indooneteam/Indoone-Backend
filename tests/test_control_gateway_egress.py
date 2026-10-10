from __future__ import annotations

import asyncio

import httpx
import pytest

from app.capabilities import control_gateway, instagram_messaging, telegram, whatsapp


SERVICE_TOKEN = "test-backend-to-gateway-service-token-123456789"


class FakeAsyncClient:
    def __init__(self, *, timeout=None, follow_redirects=None):
        self.timeout = timeout
        self.follow_redirects = follow_redirects

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None):
        self.url = url
        self.headers = headers or {}
        self.json_body = json or {}
        return httpx.Response(
            200,
            json={"messages": [{"id": "test-provider-message"}]},
            request=httpx.Request("POST", url),
        )


def _configure_gateway(monkeypatch):
    monkeypatch.setenv("INDOONE_CONTROL_GATEWAY_ORIGIN", "https://gateway.example")
    monkeypatch.setenv("INDOONE_GATEWAY_BACKEND_TOKEN", SERVICE_TOKEN)
    monkeypatch.setenv("INDOONE_GATEWAY_REQUIRED", "true")


def test_optional_legacy_mode_returns_none_when_gateway_not_configured(monkeypatch):
    monkeypatch.delenv("INDOONE_CONTROL_GATEWAY_ORIGIN", raising=False)
    monkeypatch.setenv("INDOONE_GATEWAY_REQUIRED", "false")
    assert asyncio.run(
        control_gateway.send_via_control_gateway(
            "whatsapp",
            "https://graph.facebook.com/v23.0/123/messages",
            {"Authorization": "Bearer provider-token"},
            {"to": "123", "text": "hello"},
        )
    ) is None


def test_required_gateway_fails_closed_when_origin_is_missing(monkeypatch):
    monkeypatch.delenv("INDOONE_CONTROL_GATEWAY_ORIGIN", raising=False)
    monkeypatch.setenv("INDOONE_GATEWAY_REQUIRED", "true")
    with pytest.raises(RuntimeError, match="origin is not configured"):
        asyncio.run(
            control_gateway.send_via_control_gateway(
                "whatsapp",
                "https://graph.facebook.com/v23.0/123/messages",
                {"Authorization": "Bearer provider-token"},
                {"to": "123", "text": "hello"},
            )
        )


def test_gateway_request_uses_service_token_and_provider_payload(monkeypatch):
    _configure_gateway(monkeypatch)
    calls = []

    class RecordingClient(FakeAsyncClient):
        async def post(self, url, *, headers=None, json=None):
            calls.append({"url": url, "headers": headers, "json": json})
            return await super().post(url, headers=headers, json=json)

    monkeypatch.setattr(control_gateway.httpx, "AsyncClient", RecordingClient)
    response = asyncio.run(
        control_gateway.send_via_control_gateway(
            "whatsapp",
            "https://graph.facebook.com/v23.0/123/messages",
            {"Authorization": "Bearer provider-token", "Content-Type": "application/json"},
            {"to": "123", "text": "hello"},
        )
    )
    assert response is not None and response.status_code == 200
    assert calls[0]["url"] == "https://gateway.example/internal/egress/whatsapp"
    assert calls[0]["headers"] == {"Authorization": f"Bearer {SERVICE_TOKEN}"}
    assert calls[0]["json"]["target_url"] == "https://graph.facebook.com/v23.0/123/messages"
    assert calls[0]["json"]["headers"]["Authorization"] == "Bearer provider-token"


def test_gateway_paused_reply_raises_permission_error(monkeypatch):
    _configure_gateway(monkeypatch)

    class PausedClient(FakeAsyncClient):
        async def post(self, url, *, headers=None, json=None):
            return httpx.Response(
                423,
                json={"status": "skipped", "reason": "replies_paused"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(control_gateway.httpx, "AsyncClient", PausedClient)
    with pytest.raises(PermissionError, match="replies are paused"):
        asyncio.run(
            control_gateway.send_via_control_gateway(
                "whatsapp",
                "https://graph.facebook.com/v23.0/123/messages",
                {"Authorization": "Bearer provider-token"},
                {"to": "123", "text": "hello"},
            )
        )


def test_gateway_requires_https_origin(monkeypatch):
    monkeypatch.setenv("INDOONE_CONTROL_GATEWAY_ORIGIN", "http://gateway.example")
    monkeypatch.setenv("INDOONE_GATEWAY_BACKEND_TOKEN", SERVICE_TOKEN)
    monkeypatch.setenv("INDOONE_GATEWAY_REQUIRED", "true")
    with pytest.raises(RuntimeError, match="HTTPS"):
        asyncio.run(
            control_gateway.send_via_control_gateway(
                "telegram",
                "https://api.telegram.org/bot123:ABC/sendMessage",
                {},
                {"chat_id": "1", "text": "hello"},
            )
        )


def test_whatsapp_send_adapter_uses_gateway_when_configured(monkeypatch):
    from fastapi.testclient import TestClient  # import only for type availability; no server used

    _configure_gateway(monkeypatch)
    captured = {}

    async def fake_gateway(channel, target_url, headers, payload, *, timeout=30.0):
        captured.update(channel=channel, target_url=target_url, headers=headers, payload=payload)
        return httpx.Response(
            200,
            json={"messages": [{"id": "sent-id"}]},
            request=httpx.Request("POST", "https://gateway.example/internal/egress/whatsapp"),
        )

    monkeypatch.setattr(whatsapp, "send_via_control_gateway", fake_gateway)
    monkeypatch.setenv("INDOONE_WHATSAPP_ACCESS_TOKEN", "provider-token-test")
    monkeypatch.setenv("INDOONE_WHATSAPP_PHONE_NUMBER_ID", "123456789")
    result = asyncio.run(whatsapp.send_text_message("15551234567", "hi", approved=True))
    assert result["result"]["messages"][0]["id"] == "sent-id"
    assert captured["channel"] == "whatsapp"
    assert captured["target_url"] == "https://graph.facebook.com/v23.0/123456789/messages"


def test_telegram_send_adapter_uses_gateway_when_configured(monkeypatch):
    _configure_gateway(monkeypatch)
    captured = {}

    async def fake_gateway(channel, target_url, headers, payload, *, timeout=30.0):
        captured.update(channel=channel, target_url=target_url, payload=payload)
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 7}},
            request=httpx.Request("POST", "https://gateway.example/internal/egress/telegram"),
        )

    monkeypatch.setattr(telegram, "send_via_control_gateway", fake_gateway)
    monkeypatch.setenv("INDOONE_TELEGRAM_BOT_TOKEN", "123456:ABC_test-token")
    result = asyncio.run(telegram.send_message("123", "hi", approved=True))
    assert result["message"]["message_id"] == 7
    assert captured["channel"] == "telegram"
    assert captured["target_url"] == "https://api.telegram.org/bot123456:ABC_test-token/sendMessage"


def test_instagram_message_adapter_uses_gateway_when_configured(monkeypatch):
    _configure_gateway(monkeypatch)
    captured = {}

    async def fake_access_token(user_id):
        return "instagram-provider-token"

    async def fake_gateway(channel, target_url, headers, payload, *, timeout=30.0):
        captured.update(channel=channel, target_url=target_url, payload=payload)
        return httpx.Response(
            200,
            json={"message_id": "ig-sent-id"},
            request=httpx.Request("POST", "https://gateway.example/internal/egress/instagram"),
        )

    monkeypatch.setattr(instagram_messaging, "_access_token", fake_access_token)
    monkeypatch.setattr(instagram_messaging, "send_via_control_gateway", fake_gateway)
    result = asyncio.run(
        instagram_messaging.send_text_message(
            "owner-user", "recipient-123", "hello from Indoone", approved=True
        )
    )
    assert result["sent"] is True
    assert result["result"]["message_id"] == "ig-sent-id"
    assert captured["channel"] == "instagram"
    assert captured["target_url"] == "https://graph.instagram.com/me/messages"
