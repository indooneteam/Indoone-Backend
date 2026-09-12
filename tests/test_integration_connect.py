import pytest

from app.capabilities.integrations import build_oauth_authorization


def test_oauth_authorization_requires_client_id(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="client id is not configured"):
        build_oauth_authorization("github", "a" * 16, "https://example.com/callback")


def test_oauth_authorization_contains_state_and_redirect(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    result = build_oauth_authorization("github", "state-1234567890123456", "https://example.com/callback")
    assert result["state_required"] is True
    assert result["token_storage"] == "user-scoped-server-side"
    assert result["secrets_exposed"] is False
    assert "client_id=client-123" in result["authorization_url"]
    assert "state=state-1234567890123456" in result["authorization_url"]
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in result["authorization_url"]


def test_oauth_authorization_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="integration not found"):
        build_oauth_authorization("unknown", "a" * 16, "https://example.com/callback")
