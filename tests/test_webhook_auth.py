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
    assert "instagram webhook signature is invalid" in instagram.text
    assert telegram.status_code == 401
    assert "invalid telegram webhook secret" in telegram.text
    assert all("bearer authentication required" not in response.text for response in (whatsapp, instagram, telegram))



def test_health_and_readiness_probes_are_public_but_details_remain_authenticated(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    monkeypatch.setattr("app.main.sqlite_runtime_status", lambda: {"ready": True})
    monkeypatch.setattr(
        "app.main.ai_service.ai_provider_status",
        lambda: {
            "provider": "local",
            "model": None,
            "configuration_ready": True,
            "api_key_configured": None,
        },
    )

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        details = client.get("/health/details")
        connector = client.get(
            "/api/integrations/zoho-mail/connection",
            params={"user_id": "test-user"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert details.status_code == 401
    assert connector.status_code == 401


def test_github_browser_oauth_callback_is_public_but_exchanges_state(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    captured: dict[str, str] = {}

    async def fake_exchange_oauth_code(integration_id: str, state: str, code: str) -> dict[str, object]:
        captured.update({"integration_id": integration_id, "state": state, "code": code})
        return {"integration": integration_id, "connected": True}

    monkeypatch.setattr("app.api.content.capabilities.exchange_oauth_code", fake_exchange_oauth_code)

    with TestClient(app) as client:
        response = client.get(
            "/api/integrations/github/callback",
            params={"state": "github-state-token-123456", "code": "github-auth-code"},
        )

    assert response.status_code == 200
    assert "GitHub connected to Indoone" in response.text
    assert "bearer authentication required" not in response.text.lower()
    assert captured == {
        "integration_id": "github",
        "state": "github-state-token-123456",
        "code": "github-auth-code",
    }
