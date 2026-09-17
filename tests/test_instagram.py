from urllib.parse import parse_qs, urlparse

import pytest

from app.capabilities.instagram import build_instagram_authorization


def test_instagram_authorization_uses_current_business_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_ID", "123456")
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_SECRET", "secret")

    result = build_instagram_authorization("state-token", "https://example.com/callback")
    params = parse_qs(urlparse(str(result["authorization_url"])).query)

    assert params["client_id"] == ["123456"]
    assert params["response_type"] == ["code"]
    assert params["state"] == ["state-token"]
    assert params["scope"] == ["instagram_business_basic,instagram_business_content_publish"]
    assert result["professional_accounts_only"] is True


def test_instagram_authorization_can_request_messages_and_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_ID", "123456")
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_SECRET", "secret")

    result = build_instagram_authorization(
        "state-token",
        "https://example.com/callback",
        include_publishing=False,
        include_messages=True,
        include_comments=True,
    )
    params = parse_qs(urlparse(str(result["authorization_url"])).query)

    assert params["scope"] == [
        "instagram_business_basic,instagram_business_manage_messages,instagram_business_manage_comments"
    ]
