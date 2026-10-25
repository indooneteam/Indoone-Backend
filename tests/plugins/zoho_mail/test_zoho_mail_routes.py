from __future__ import annotations

from pathlib import Path
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
    assert payload["permissions_complete"] is False
    assert payload["reauthorization_required"] is True
    assert "ZohoMail.messages.DELETE" in payload["missing_scopes"]
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
    from app.capabilities.connectors.integrations import get_integration

    integration = get_integration("zoho_mail")
    assert integration is not None
    assert integration["name"] == "Zoho Mail"
    assert integration["auth_type"] == "oauth2"
    assert "ZohoMail.messages.READ" in integration["permissions"]


def test_zoho_mail_reply_endpoint_calls_provider_service(monkeypatch) -> None:
    from app.plugins.zoho_mail import routes as zoho_routes

    async def fake_reply(
        user_id: str,
        account_id: str,
        message_id: str,
        from_address: str,
        to_address: str,
        subject: str,
        content: str,
        approved: bool = False,
    ) -> dict[str, object]:
        assert approved is True
        return {
            "status": {"code": 200, "description": "success"},
            "secrets_exposed": False,
            "message_id": message_id,
        }

    monkeypatch.setattr(zoho_routes, "reply_to_message", fake_reply)
    client = _client(monkeypatch)
    response = client.post(
        "/api/integrations/zoho-mail/reply",
        json={
            "user_id": "test-user",
            "account_id": "account-1",
            "message_id": "message-1",
            "from_address": "user@indoone.example",
            "to_address": "sender@example.com",
            "subject": "Re: Hello",
            "content": "Thanks for your email.",
            "approved": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["message_id"] == "message-1"
    assert response.json()["secrets_exposed"] is False



def test_zoho_mail_oauth_error_callback_without_code_is_handled_without_bearer_auth(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    client = TestClient(app, raise_server_exceptions=False)

    # Zoho OAuth denial callbacks include error/state and omit code.
    response = client.get(
        "/api/integrations/zoho-mail/callback",
        params={"error": "access_denied", "state": "verification-check"},
    )

    assert response.status_code == 400
    assert response.status_code != 401
    assert "connection was not completed" in response.text


def test_zoho_mail_callback_missing_code_and_state_returns_clear_400(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/integrations/zoho-mail/callback")

    assert response.status_code == 400
    assert "authorization code or state" in response.json()["message"]



def test_zoho_mail_send_requires_explicit_approval(monkeypatch) -> None:
    from app.plugins.zoho_mail import routes as zoho_routes

    async def fake_send(*args, approved=False, **kwargs):
        if not approved:
            raise PermissionError("explicit approval is required for Zoho Mail send operations")
        return {"status": {"code": 200, "description": "success"}}

    monkeypatch.setattr(zoho_routes, "send_message", fake_send)
    client = _client(monkeypatch)
    response = client.post(
        "/api/integrations/zoho-mail/send",
        json={
            "user_id": "test-user",
            "account_id": "account-1",
            "from_address": "user@example.com",
            "to_address": "friend@example.com",
            "subject": "Hello",
            "content": "Hi",
        },
    )
    assert response.status_code == 403


def test_zoho_mail_disconnect_when_already_disconnected(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false")
    monkeypatch.setattr(zoho_oauth, "get_integration_token", lambda user_id, integration_id: None)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/integrations/zoho-mail/disconnect",
        json={"user_id": "test-user"},
    )
    assert response.status_code == 200
    assert response.json()["connected"] is False
    assert response.json()["secrets_exposed"] is False

def test_zoho_default_scope_includes_delete_permission() -> None:
    required_scope = "ZohoMail.messages.DELETE"
    assert required_scope in zoho_oauth._DEFAULT_SCOPE.split(",")

    example_env = Path(__file__).resolve().parents[3] / ".env.example"
    assert required_scope in example_env.read_text(encoding="utf-8")

def test_zoho_connection_status_reports_complete_permissions(monkeypatch) -> None:
    complete_scope = ",".join(zoho_oauth.ZOHO_MAIL_REQUIRED_SCOPES)
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {
            "user_id": user_id,
            "integration_id": integration_id,
            "scope": complete_scope,
            "updated_at": "2026-10-11T12:00:00+00:00",
        },
    )
    response = _client(monkeypatch).get(
        "/api/integrations/zoho-mail/connection",
        params={"user_id": "test-user"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["permissions_complete"] is True
    assert payload["missing_scopes"] == []
    assert payload["reauthorization_required"] is False



def test_zoho_authorization_always_requests_all_required_scopes(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_ID", "zoho-public-client-id")
    monkeypatch.setenv("INDOONE_ZOHO_CLIENT_SECRET", "server-only-client-secret")
    monkeypatch.setenv(
        "INDOONE_ZOHO_REDIRECT_URI",
        "https://indoone.example/api/integrations/zoho-mail/callback",
    )
    # Simulate an old deployment override that included only a subset of permissions.
    monkeypatch.setenv(
        "INDOONE_ZOHO_SCOPE",
        "ZohoMail.accounts.READ,ZohoMail.messages.READ",
    )
    monkeypatch.setattr(zoho_oauth, "create_oauth_state", lambda *_args: None)

    result = zoho_oauth.start_zoho_mail_authorization("test-user")
    query = parse_qs(urlparse(str(result["authorization_url"])).query)
    requested_scopes = set(query["scope"][0].split(","))

    assert requested_scopes == set(zoho_oauth.ZOHO_MAIL_REQUIRED_SCOPES)
    assert zoho_oauth.zoho_mail_auth_config()["scope"] == zoho_oauth._DEFAULT_SCOPE


def test_zoho_connection_status_flags_each_missing_required_permission(monkeypatch) -> None:
    monkeypatch.setattr(
        oauth_routes,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {
            "user_id": user_id,
            "integration_id": integration_id,
            "scope": "ZohoMail.accounts.READ,ZohoMail.messages.READ",
            "updated_at": "2026-10-11T12:00:00+00:00",
        },
    )
    response = _client(monkeypatch).get(
        "/api/integrations/zoho-mail/connection",
        params={"user_id": "test-user"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["permissions_complete"] is False
    assert set(payload["missing_scopes"]) == {
        "ZohoMail.folders.READ",
        "ZohoMail.messages.CREATE",
        "ZohoMail.messages.DELETE",
    }
    assert payload["reauthorization_required"] is True
