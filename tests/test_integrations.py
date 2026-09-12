from fastapi.testclient import TestClient

from app.capabilities.integrations import build_oauth_authorization
from app.main import app


def test_integrations_list_is_safe_metadata(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_SLACK_CLIENT_ID", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/integrations")
    assert response.status_code == 200
    items = response.json()["integrations"]
    assert {item["id"] for item in items} >= {"github", "google_calendar", "slack"}
    assert all(item["configured"] is False for item in items)
    assert all("token" not in item and "secret" not in item for item in items)


def test_integration_lookup_and_missing_provider() -> None:
    with TestClient(app) as client:
        response = client.get("/api/integrations/github")
        missing = client.get("/api/integrations/not-configured")
    assert response.status_code == 200
    assert response.json()["integration"]["auth_type"] == "oauth2"
    assert "repo:read" in response.json()["integration"]["permissions"]
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


def test_integration_status_is_user_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.get("/api/integrations/github/status", params={"user_id": "user-123"})
    assert response.status_code == 200
    assert response.json()["connected"] is False
    assert response.json()["metadata"] is None
