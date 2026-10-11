from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.plugins.gmail import oauth_routes


def test_gmail_oauth_endpoints_are_mounted_and_reachable(monkeypatch) -> None:
    # Exercise the real FastAPI routing stack without requiring production credentials.
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false")
    monkeypatch.setattr(
        oauth_routes,
        "gmail_server_authorization_config",
        lambda: {
            "integration": "gmail",
            "server_client_id": "public-client-id",
            "offline_access": True,
        },
    )
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: None,
    )

    async def fake_exchange(user_id, server_auth_code, requested_scope=""):
        return {
            "integration": "gmail",
            "user_id": user_id,
            "connected": True,
            "provider_ok": True,
            "scope": requested_scope,
            "summary": {"email_address": "gmail-test@example.com"},
            "secrets_exposed": False,
        }

    async def fake_store(user_id, integration_id, access_token, scope="", expires_at=None):
        return {
            "integration": integration_id,
            "user_id": user_id,
            "connected": True,
            "summary": {"email_address": "gmail-test@example.com"},
            "secrets_exposed": False,
        }

    monkeypatch.setattr(oauth_routes, "exchange_gmail_server_auth_code", fake_exchange)
    monkeypatch.setattr(oauth_routes, "store_native_access_token", fake_store)

    # Avoid running the app lifespan: this is a route-wiring test, not a deployment test.
    client = TestClient(app, raise_server_exceptions=False)

    auth_config = client.get("/api/integrations/gmail/auth-config")
    assert auth_config.status_code == 200
    assert auth_config.json()["server_client_id"] == "public-client-id"

    server_auth = client.post(
        "/api/integrations/gmail/server-auth-code",
        json={
            "user_id": "user-one",
            "server_auth_code": "test-code",
            "scope": "https://www.googleapis.com/auth/gmail.modify",
        },
    )
    assert server_auth.status_code == 200
    assert server_auth.json()["connected"] is True

    native_token = client.post(
        "/api/integrations/gmail/native-token",
        json={
            "user_id": "user-one",
            "access_token": "test-access-token",
            "scope": "https://www.googleapis.com/auth/gmail.modify",
        },
    )
    assert native_token.status_code == 200
    assert native_token.json()["connected"] is True

    status = client.get("/api/integrations/gmail/status", params={"user_id": "user-one"})
    assert status.status_code == 200
    assert status.json()["connected"] is False
    assert status.json()["metadata"] == {}


@pytest.mark.asyncio
async def test_gmail_status_is_disconnected_when_no_token_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: None,
    )

    result = await oauth_routes.gmail_connection_status("user-one")

    assert result["integration"] == "gmail"
    assert result["user_id"] == "user-one"
    assert result["connected"] is False
    assert result["metadata"] == {}
    assert result["secrets_exposed"] is False


@pytest.mark.asyncio
async def test_gmail_status_exposes_metadata_but_not_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {
            "user_id": user_id,
            "integration_id": integration_id,
            "access_token": b"do-not-return-access-token",
            "refresh_token": b"do-not-return-refresh-token",
            "scope": "https://www.googleapis.com/auth/gmail.modify",
            "expires_at": "2026-10-11T14:00:00+00:00",
            "updated_at": "2026-10-11T13:00:00+00:00",
        },
    )

    result = await oauth_routes.gmail_connection_status("user-one")

    assert result["connected"] is True
    assert result["metadata"]["scope"] == "https://www.googleapis.com/auth/gmail.modify"
    assert "access_token" not in result
    assert "refresh_token" not in result
    assert "do-not-return-access-token" not in str(result)
    assert "do-not-return-refresh-token" not in str(result)
