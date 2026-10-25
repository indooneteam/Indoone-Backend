import pytest
from fastapi.testclient import TestClient

from app.capabilities.connectors.integrations import build_oauth_authorization
from app.api.content import capabilities as capabilities_api
from app.main import app


def test_integrations_list_is_safe_metadata(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_SLACK_CLIENT_ID", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/integrations")
    assert response.status_code == 200
    items = response.json()["integrations"]
    assert {item["id"] for item in items} >= {"github", "zoho_mail", "slack"}
    assert all(item["configured"] is False for item in items)
    assert all("token" not in item and "secret" not in item for item in items)


def test_integration_lookup_and_missing_provider() -> None:
    with TestClient(app) as client:
        response = client.get("/api/integrations/github")
        missing = client.get("/api/integrations/not-configured")
    assert response.status_code == 200
    assert response.json()["integration"]["auth_type"] == "oauth2"
    assert "public_repo" in response.json()["integration"]["permissions"]
    assert missing.status_code == 404


def test_oauth_contract_requires_client_id(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    try:
        build_oauth_authorization("github", "a" * 32, "https://app.example/callback")
    except RuntimeError as exc:
        assert "client id" in str(exc)
    else:
        raise AssertionError("missing client id must fail")


def test_oauth_contract_builds_user_scoped_authorization(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    result = build_oauth_authorization("github", "state-token-1234567890", "https://app.example/callback")
    assert "client_id=client-123" in result["authorization_url"]
    assert "state=state-token-1234567890" in result["authorization_url"]
    assert result["token_storage"] == "user-scoped-encrypted-server-side"
    assert result["secrets_exposed"] is False


def test_integration_connect_endpoint_creates_user_bound_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/connect",
            json={"user_id": "user-123", "redirect_uri": "https://app.example/callback"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["integration"] == "github"
    assert payload["state_required"] is True
    assert payload["secrets_exposed"] is False
    assert len(payload["state"]) >= 32


def test_integration_connect_endpoint_rejects_unknown_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/unknown/connect",
            json={"user_id": "user-123", "redirect_uri": "https://app.example/callback"},
        )
    assert response.status_code == 400


def test_integration_callback_requires_code_and_state() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/callback",
            json={"state": "short"},
        )
    assert response.status_code == 422


def test_github_disconnect_removes_only_the_current_users_token(tmp_path, monkeypatch) -> None:
    from cryptography.fernet import Fernet
    from app.capabilities.connectors.store import get_integration_token_metadata, upsert_integration_token

    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    cipher = Fernet(Fernet.generate_key())
    upsert_integration_token("user-one", "github", cipher.encrypt(b"one-token"), None, "Bearer", "public_repo read:user", None)
    upsert_integration_token("user-two", "github", cipher.encrypt(b"two-token"), None, "Bearer", "public_repo read:user", None)

    with TestClient(app) as client:
        response = client.post("/api/integrations/github/disconnect", json={"user_id": "user-one"})

    assert response.status_code == 200
    assert response.json()["disconnected"] is True
    assert response.json()["was_connected"] is True
    assert get_integration_token_metadata("user-one", "github") is None
    assert get_integration_token_metadata("user-two", "github") is not None


def test_integration_status_is_user_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.get("/api/integrations/github/status", params={"user_id": "user-123"})
    assert response.status_code == 200
    assert response.json()["connected"] is False
    assert response.json()["metadata"] is None



def test_github_oauth_uses_supported_least_privilege_scopes(monkeypatch) -> None:
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "github-public-client-id")
    result = build_oauth_authorization(
        "github", "github-test-state", "https://indoone.example/api/integrations/github/callback"
    )
    query = parse_qs(urlsplit(str(result["authorization_url"])).query)

    assert set(query["scope"][0].split()) == {"read:user", "public_repo"}
    assert result["secrets_exposed"] is False


def test_github_plugin_routes_keep_existing_api_paths() -> None:
    with TestClient(app) as client:
        repositories = client.post(
            "/api/integrations/github/repositories", json={"user_id": "route-check-user"}
        )
        issues = client.post(
            "/api/integrations/github/issues", json={"user_id": "route-check-user"}
        )
        pull_requests = client.post(
            "/api/integrations/github/pull-requests",
            json={"user_id": "route-check-user", "repository": "owner/repo"},
        )
        create_issue = client.post(
            "/api/integrations/github/create-issue",
            json={"user_id": "route-check-user", "repository": "owner/repo", "title": "test"},
        )
        comment = client.post(
            "/api/integrations/github/comment",
            json={"user_id": "route-check-user", "repository": "owner/repo", "number": 1, "body": "test"},
        )

    # The expected API-level validation/auth responses prove these plugin routes
    # are mounted without requiring a live GitHub OAuth token.
    assert repositories.status_code == 400
    assert issues.status_code == 400
    assert pull_requests.status_code == 400
    assert create_issue.status_code == 403
    assert comment.status_code == 403


def test_github_browser_oauth_callback_shows_github_success(monkeypatch) -> None:
    async def fake_exchange(integration_id: str, state: str, code: str) -> dict[str, object]:
        assert integration_id == "github"
        assert state == "github-test-state-123456"
        assert code == "sample-code"
        return {"integration": integration_id, "connected": True}

    monkeypatch.setattr(capabilities_api, "exchange_oauth_code", fake_exchange)
    with TestClient(app) as client:
        response = client.get(
            "/api/integrations/github/callback",
            params={"state": "github-test-state-123456", "code": "sample-code"},
        )

    assert response.status_code == 200
    assert "GitHub connected to Indoone" in response.text
    assert "Zoho Mail connected to Indoone" not in response.text
