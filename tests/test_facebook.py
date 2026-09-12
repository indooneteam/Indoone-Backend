from __future__ import annotations

from app.capabilities.facebook import build_facebook_authorization


def test_facebook_authorization_uses_pages_scopes(monkeypatch):
    monkeypatch.setenv("INDOONE_FACEBOOK_APP_ID", "facebook-app")
    monkeypatch.setenv("INDOONE_FACEBOOK_APP_SECRET", "facebook-secret")
    result = build_facebook_authorization("state-123", "https://example.com/callback")
    assert result["integration"] == "facebook"
    assert "pages_show_list" in result["scope"]
    assert "pages_read_engagement" in result["scope"]
    assert "pages_manage_posts" in result["scope"]
    assert result["pages_only"] is True
    assert result["secrets_exposed"] is False
