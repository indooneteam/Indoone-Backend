from __future__ import annotations

import pytest

import app.capabilities.youtube_playlist_manager as manager


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        return self.response


def test_validate_playlist_fields() -> None:
    assert manager._validate_playlist_fields(title="My List", description="Desc", privacy_status="PUBLIC") == (
        "My List",
        "Desc",
        "public",
    )
    with pytest.raises(ValueError, match="150 characters"):
        manager._validate_playlist_fields(title="x" * 151)


def test_playlist_writes_require_explicit_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        pytest.raises(AssertionError)
        # direct coroutine checks keep authorization before token access
    with pytest.raises(PermissionError, match="explicit approval"):
        import asyncio
        asyncio.run(manager.create_playlist("user-1", "List"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(manager.update_playlist("user-1", "playlist-1", title="New"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(manager.delete_playlist("user-1", "playlist-1"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(manager.add_video_to_playlist("user-1", "playlist-1", "video-1"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(manager.remove_playlist_item("user-1", "item-1"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(manager.reorder_playlist_item("user-1", "item-1", 0))


@pytest.mark.asyncio
async def test_list_playlists_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse({"items": [{"id": "pl-1"}], "nextPageToken": "next", "pageInfo": {"totalResults": 1}}))
    monkeypatch.setattr(manager, "_manager_token", fake_manager_token)
    monkeypatch.setattr(manager.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await manager.list_playlists("user-1", max_results=10, page_token="page")

    assert result["playlists"] == [{"id": "pl-1"}]
    assert fake_client.calls[0][0] == "GET"
    assert fake_client.calls[0][1].endswith("/playlists")
    assert fake_client.calls[0][2]["params"] == {"part": "snippet,contentDetails,status", "mine": "true", "maxResults": 10, "pageToken": "page"}


@pytest.mark.asyncio
async def test_create_playlist_sends_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse({"id": "pl-1"}))
    monkeypatch.setattr(manager, "_manager_token", fake_manager_token)
    monkeypatch.setattr(manager.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await manager.create_playlist("user-1", "My List", "Desc", "public", approved=True)

    assert result["operation"] == "create_playlist"
    call = fake_client.calls[0][2]
    assert call["params"] == {"part": "snippet,status"}
    assert call["json"] == {
        "snippet": {"title": "My List", "description": "Desc"},
        "status": {"privacyStatus": "public"},
    }


@pytest.mark.asyncio
async def test_add_video_includes_optional_position(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse({"id": "item-1"}))
    monkeypatch.setattr(manager, "_manager_token", fake_manager_token)
    monkeypatch.setattr(manager.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await manager.add_video_to_playlist("user-1", "pl-1", "video-1", 2, approved=True)

    assert result["item"] == {"id": "item-1"}
    assert fake_client.calls[0][2]["json"] == {
        "snippet": {
            "playlistId": "pl-1",
            "resourceId": {"kind": "youtube#video", "videoId": "video-1"},
            "position": 2,
        }
    }
