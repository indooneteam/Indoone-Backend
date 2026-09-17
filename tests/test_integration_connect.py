import pytest
from cryptography.fernet import Fernet

from app.capabilities.integrations import build_oauth_authorization, exchange_oauth_code
from app.capabilities.store import create_oauth_state, get_integration_token_metadata


def test_oauth_authorization_requires_client_id(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="client id is not configured"):
        build_oauth_authorization("github", "a" * 16, "https://example.com/callback")


def test_oauth_authorization_contains_state_and_redirect(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    result = build_oauth_authorization("github", "state-1234567890123456", "https://example.com/callback")
    assert result["state_required"] is True
    assert result["token_storage"] == "user-scoped-encrypted-server-side"
    assert result["secrets_exposed"] is False
    assert "client_id=client-123" in result["authorization_url"]
    assert "state=state-1234567890123456" in result["authorization_url"]
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in result["authorization_url"]


def test_oauth_authorization_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="integration not found"):
        build_oauth_authorization("unknown", "a" * 16, "https://example.com/callback")


@pytest.mark.asyncio
async def test_oauth_code_exchange_is_user_scoped_and_redacted(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "capabilities.db"
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(db_path))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_SECRET", "secret-123")
    state = "state-for-user-123456789"
    create_oauth_state(state, "user-123", "github", "https://example.com/callback")

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"access_token": "access-secret", "token_type": "Bearer", "scope": "repo:read", "expires_in": 3600}

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs):
            assert kwargs["data"]["client_id"] == "client-123"
            assert kwargs["data"]["client_secret"] == "secret-123"
            return Response()

    monkeypatch.setattr("app.capabilities.integrations.httpx.AsyncClient", Client)
    result = await exchange_oauth_code("github", state, "auth-code")
    assert result["connected"] is True
    assert result["user_id"] == "user-123"
    assert result["secrets_exposed"] is False
    assert "access-secret" not in str(result)
    metadata = get_integration_token_metadata("user-123", "github")
    assert metadata is not None
    assert metadata["scope"] == "repo:read"
    assert "access_token" not in metadata
    assert get_integration_token_metadata("user-999", "github") is None


def test_oauth_state_is_single_use_and_explicitly_user_bound(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    from app.capabilities.store import consume_oauth_state

    state = "single-use-state-123456"
    create_oauth_state(state, "user-abc", "slack", "https://example.com/callback")
    first = consume_oauth_state(state, "slack")
    second = consume_oauth_state(state, "slack")
    assert first == {"user_id": "user-abc", "integration_id": "slack", "redirect_uri": "https://example.com/callback"}
    assert second is None
