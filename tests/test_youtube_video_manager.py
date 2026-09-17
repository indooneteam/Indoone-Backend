from __future__ import annotations

import pytest

import app.capabilities.youtube_video_manager as manager


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, *, response: FakeResponse, expected_method: str, expected_url: str) -> None:
        self.response = response
        self.expected_method = expected_method
        self.expected_url = expected_url
        self.calls: list[dict[str, object]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def put(self, url: str, **kwargs: object) -> FakeResponse:
        assert self.expected_method == "put"
        assert url == self.expected_url
        self.calls.append(kwargs)
        return self.response

    async def delete(self, url: str, **kwargs: object) -> FakeResponse:
        assert self.expected_method == "delete"
        assert url == self.expected_url
        self.calls.append(kwargs)
        return self.response

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        assert self.expected_method == "post"
        assert url == self.expected_url
        self.calls.append(kwargs)
        return self.response


def test_manager_authorization_requests_full_youtube_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    result = manager.build_youtube_manager_authorization("state-value", "https://example.com/callback")
    assert result["connect_mode"] == "channel_manager"
    assert result["scope"] == "https://www.googleapis.com/auth/youtube"
    assert "state=state-value" in str(result["authorization_url"])
    assert "delete videos with explicit approval" in result["capabilities"]


def test_manager_scope_requires_full_channel_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "get_integration_token", lambda user_id, integration_id: {
        "scope": "https://www.googleapis.com/auth/youtube.upload",
    })
    with pytest.raises(PermissionError, match="video manager access is not connected"):
        manager._manager_scope("user-1")


def test_build_video_update_payload_preserves_existing_fields() -> None:
    parts, payload = manager.build_video_update_payload(
        {
            "id": "video-1",
            "snippet": {
                "title": "Old",
                "description": "Old description",
                "categoryId": "22",
                "tags": ["one", "two"],
            },
            "status": {"privacyStatus": "private"},
        },
        description="New description",
        privacy_status="unlisted",
    )
    assert parts == ["snippet", "status"]
    assert payload["id"] == "video-1"
    assert payload["snippet"] == {
        "title": "Old",
        "categoryId": "22",
        "description": "New description",
        "tags": ["one", "two"],
    }
    assert payload["status"] == {"privacyStatus": "unlisted"}


@pytest.mark.asyncio
async def test_update_video_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        assert user_id == "user-1"
        return "token"

    async def fake_get_videos(video_ids: list[str], user_id: str) -> dict[str, object]:
        assert video_ids == ["video-1"]
        assert user_id == "user-1"
        return {
            "videos": [{
                "id": "video-1",
                "snippet": {"title": "Old", "description": "Old description", "categoryId": "22"},
                "status": {"privacyStatus": "private"},
            }]
        }

    fake_client = FakeClient(
        response=FakeResponse({"id": "video-1", "snippet": {"title": "New"}, "status": {"privacyStatus": "public"}}),
        expected_method="put",
        expected_url="https://www.googleapis.com/youtube/v3/videos",
    )
    monkeypatch.setattr(manager, "_manager_token", fake_manager_token)
    monkeypatch.setattr(manager, "get_videos", fake_get_videos)
    monkeypatch.setattr(manager.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await manager.update_video("user-1", "video-1", title="New", privacy_status="public", approved=True)

    assert result["operation"] == "update_video"
    assert result["video"]["id"] == "video-1"
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["params"] == {"part": "snippet,status"}
    assert call["headers"] == {"Authorization": "Bearer token", "Accept": "application/json"}
    assert call["json"] == {
        "id": "video-1",
        "snippet": {"title": "New", "categoryId": "22", "description": "Old description"},
        "status": {"privacyStatus": "public"},
    }


@pytest.mark.asyncio
async def test_delete_video_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(
        response=FakeResponse(status_code=204),
        expected_method="delete",
        expected_url="https://www.googleapis.com/youtube/v3/videos",
    )
    monkeypatch.setattr(manager, "_manager_token", fake_manager_token)
    monkeypatch.setattr(manager.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await manager.delete_video("user-1", "video-1", approved=True)

    assert result["deleted"] is True
    assert result["video_id"] == "video-1"
    assert fake_client.calls[0]["params"] == {"id": "video-1"}


@pytest.mark.asyncio
async def test_thumbnail_rejects_unsupported_mime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "_manager_token", lambda user_id: None)
    with pytest.raises(ValueError, match="image/jpeg or image/png"):
        await manager.set_video_thumbnail("user-1", "video-1", b"image", "image/gif", approved=True)


@pytest.mark.asyncio
async def test_writes_require_explicit_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        await manager.update_video("user-1", "video-1", title="New")
    with pytest.raises(PermissionError, match="explicit approval"):
        await manager.delete_video("user-1", "video-1")
    with pytest.raises(PermissionError, match="explicit approval"):
        await manager.set_video_thumbnail("user-1", "video-1", b"image", "image/png")
