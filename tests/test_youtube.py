from __future__ import annotations

import pytest

import app.capabilities.youtube as youtube
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
    assert "view the authenticated user's YouTube channel dashboard" not in normal
    assert "list the authenticated user's uploaded videos" not in normal
    assert "view the authenticated user's YouTube channel dashboard" in channel
    assert "list the authenticated user's uploaded videos" in channel
    assert "upload videos with explicit approval" in channel


def test_youtube_capabilities_reject_unknown_mode() -> None:
    with pytest.raises(ValueError, match="connect_mode must be normal or channel"):
        youtube_connect_capabilities("other")


@pytest.mark.asyncio
async def test_youtube_channel_dashboard_combines_channel_and_uploaded_videos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_my_channel(user_id: str) -> dict[str, object]:
        assert user_id == "user-1"
        return {"channels": [{"id": "channel-1", "snippet": {"title": "Indoone"}}]}

    async def fake_list_my_videos(user_id: str, max_results: int = 20, page_token: str = "") -> dict[str, object]:
        assert user_id == "user-1"
        assert max_results == 5
        assert page_token == "next"
        return {
            "videos": [{"id": "video-1", "title": "Hello"}],
            "next_page_token": "next-2",
            "total_results": 42,
        }

    monkeypatch.setattr(youtube, "get_my_channel", fake_get_my_channel)
    monkeypatch.setattr(youtube, "list_my_videos", fake_list_my_videos)

    result = await youtube.get_channel_dashboard("user-1", max_results=5, page_token="next")

    assert result["integration"] == "youtube"
    assert result["channel"] == {"id": "channel-1", "snippet": {"title": "Indoone"}}
    assert result["videos"] == [{"id": "video-1", "title": "Hello"}]
    assert result["next_page_token"] == "next-2"
    assert result["total_results"] == 42
    assert result["secrets_exposed"] is False


def test_youtube_dashboard_requires_safe_result_count() -> None:
    with pytest.raises(ValueError, match="max_results must be between 1 and 50"):
        # The function validates before making any network call.
        import asyncio

        with pytest.raises(ValueError, match="max_results must be between 1 and 50"):
            asyncio.run(youtube.get_channel_dashboard("user-1", max_results=0))
