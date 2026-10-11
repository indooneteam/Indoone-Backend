from __future__ import annotations

import pytest

from app.main import app
from app.plugins.gmail import oauth_routes


def test_gmail_oauth_routes_are_mounted_under_api_prefix() -> None:
    registered = {
        (route.path, method)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }
    expected = {
        ("/api/integrations/gmail/auth-config", "GET"),
        ("/api/integrations/gmail/server-auth-code", "POST"),
        ("/api/integrations/gmail/native-token", "POST"),
        ("/api/integrations/gmail/status", "GET"),
    }
    assert expected <= registered


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
