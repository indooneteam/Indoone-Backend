from __future__ import annotations

import pytest

from app.capabilities.youtube import build_youtube_authorization, youtube_connect_capabilities


def test_youtube_authorization_uses_channel_scope_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = build_youtube_authorization("state-value", "https://example.com/callback")
    assert result["integration"] == "youtube"
    assert result["connect_mode"] == "channel"
    assert "youtube.upload" in str(result["scope"])
    assert "state=state-value" in str(result["authorization_url"])
    assert "upload videos with explicit approval" in result["capabilities"]


def test_youtube_normal_connect_uses_read_only_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = build_youtube_authorization(
        "state-value",
        "https://example.com/callback",
        connect_mode="normal",
    )
    assert result["connect_mode"] == "normal"
    assert "youtube.readonly" in str(result["scope"])
    assert "upload videos with explicit approval" not in result["capabilities"]


def test_youtube_read_only_backward_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = build_youtube_authorization(
        "state-value",
        "https://example.com/callback",
        read_only=True,
    )
    assert result["connect_mode"] == "normal"
    assert "youtube.readonly" in str(result["scope"])


def test_youtube_capabilities_are_separated_by_mode() -> None:
    normal = youtube_connect_capabilities("normal")
    channel = youtube_connect_capabilities("channel")
    assert "search videos, channels, and playlists" in normal
    assert "get public video details and statistics" in normal
    assert "view the authenticated user's YouTube channel" not in normal
    assert "upload videos with explicit approval" not in normal
    assert "view the authenticated user's YouTube channel" in channel
    assert "upload videos with explicit approval" in channel


def test_youtube_capabilities_reject_unknown_mode() -> None:
    with pytest.raises(ValueError, match="connect_mode must be normal or channel"):
        youtube_connect_capabilities("other")
