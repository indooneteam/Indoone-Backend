import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_provider_webhooks_use_provider_validation_without_app_bearer(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_WHATSAPP_APP_SECRET", "test-whatsapp-secret")
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_SECRET", "test-instagram-secret")
    monkeypatch.setenv("INDOONE_TELEGRAM_WEBHOOK_SECRET", "test-telegram-secret")

    with TestClient(app) as client:
        whatsapp = client.post(
            "/api/integrations/whatsapp/webhook",
            json={},
            headers={"X-Hub-Signature-256": "sha256=invalid"},
        )
        instagram = client.post(
            "/api/integrations/instagram/webhook",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=invalid",
            },
        )
        telegram = client.post(
            "/api/telegram/webhook/incoming",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )

    assert whatsapp.status_code == 403
    assert "invalid whatsapp webhook signature" in whatsapp.text
    assert instagram.status_code == 403
    assert "invalid instagram webhook signature" in instagram.text
    assert telegram.status_code == 401
    assert "invalid telegram webhook secret" in telegram.text
    assert all("bearer authentication required" not in response.text for response in (whatsapp, instagram, telegram))
