from __future__ import annotations

import pytest

from app.capabilities.youtube import build_youtube_authorization


def test_youtube_authorization_uses_upload_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = build_youtube_authorization("state-value", "https://example.com/callback")
    assert result["integration"] == "youtube"
    assert "youtube.upload" in str(result["scope"])
    assert "state=state-value" in str(result["authorization_url"])


def test_youtube_authorization_can_request_read_only_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = build_youtube_authorization("state-value", "https://example.com/callback", read_only=True)
    assert "youtube.readonly" in str(result["scope"])
