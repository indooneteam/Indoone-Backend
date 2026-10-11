from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.main import app
from app.plugins.zoho_mail import oauth as zoho_oauth
from app.plugins.zoho_mail import oauth_routes


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false")
    return TestClient(app, raise_server_exceptions=False)


def test_zoho_mail_auth_config_never_returns_client_secret(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_ID", "zoho-public-client-id")
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_SECRET", "server-only-client-secret")
    monkeypatch.setenv("INDOONE_ZOHO_REDIRECT_URI", "https://indoone.example/api/integrations/zoho-mail/callback")
    client = _client(monkeypatch)

    response = client.get("/api/integrations/zoho-mail/auth-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["client_id_configured"] is True
    assert payload["client_secret_configured"] is True
    assert payload["secrets_exposed"] is False
    assert "server-only-client-secret" not in response.text
    assert "client_secret" not in payload


def test_zoho_mail_start_creates_user_scoped_state_and_returns_authorization_url(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_ID", "zoho-public-client-id")
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_SECRET", "server-only-client-secret")
    monkeypatch.setenv("INDOONE_ZOHO_REDIRECT_URI", "https://indoone.example/api/integrations/zoho-mail/callback")
    saved: dict[str, str] = {}

    def fake_create_state(state: str, user_id: str, integration_id: str, redirect_uri: str) -> None:
        saved.update(state=state, user_id=user_id, integration_id=integration_id, redirect_uri=redirect_uri)

    monkeypatch.setattr(zoho_oauth, "create_oauth_state", fake_create_state)
    client = _client(monkeypatch)

    response = client.post("/api/integrations/zoho-mail/start", json={"user_id": "test-user"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["secrets_exposed"] is False
    auth_url = urlparse(payload["authorization_url"])
    params = parse_qs(auth_url.query)
    assert auth_url.scheme == "https"
    assert auth_url.netloc == "accounts.zoho.in"
    assert params["client_id"] == ["zoho-public-client-id"]
    assert params["access_type"] == ["offline"]
    assert params["state"] == [saved["state"]]
    assert saved["user_id"] == "test-user"
    assert saved["integration_id"] == "zoho_mail"
    assert "server-only-client-secret" not in response.text


def test_zoho_mail_status_does_not_expose_saved_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {
            "user_id": user_id,
            "integration_id": integration_id,
            "access_token": b"private-access-token",
            "refresh_token": b"private-refresh-token",
            "scope": "ZohoMail.accounts.READ",
            "updated_at": "2026-10-11T12:00:00+00:00",
        },
    )
    client = _client(monkeypatch)

    response = client.get("/api/integrations/zoho-mail/connection", params={"user_id": "test-user"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["scope"] == "ZohoMail.accounts.READ"
    assert payload["secrets_exposed"] is False
    assert "access_token" not in payload
    assert "refresh_token" not in payload
    assert "private-access-token" not in response.text
    assert "private-refresh-token" not in response.text


def test_zoho_mail_oauth_routes_are_mounted(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/integrations/zoho-mail/connection", params={"user_id": "test-user"})
    assert response.status_code == 200
    assert response.json()["integration"] == "zoho_mail"
    assert response.json()["secrets_exposed"] is False


def test_zoho_mail_generic_integration_registry_supports_oauth() -> None:
    from app.capabilities.integrations import get_integration

    integration = get_integration("zoho_mail")
    assert integration is not None
    assert integration["name"] == "Zoho Mail"
    assert integration["auth_type"] == "oauth2"
    assert "ZohoMail.messages.READ" in integration["permissions"]
