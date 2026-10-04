from __future__ import annotations

from starlette.requests import Request

import app.integrations.whatsapp.api as whatsapp_api


async def test_whatsapp_webhook_verification_returns_plain_text(monkeypatch):
    monkeypatch.setenv("INDOONE_WHATSAPP_WEBHOOK_VERIFY_TOKEN", "test-token")

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/integrations/whatsapp/webhook",
            "query_string": b"hub.mode=subscribe&hub.verify_token=test-token&hub.challenge=123456",
            "headers": [],
            "scheme": "http",
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        }
    )

    response = await whatsapp_api.verify(request)

    assert response.status_code == 200
    assert response.media_type == "text/plain"
    assert response.body == b"123456"
