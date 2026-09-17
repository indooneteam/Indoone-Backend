from __future__ import annotations

import asyncio

import pytest

import app.capabilities.youtube_comments as comments


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code
        self.content = b"{}" if status_code != 204 else b""

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


def test_comment_write_operations_require_explicit_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(comments.create_top_level_comment("user-1", "Hello", video_id="video-1", channel_id="channel-1"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(comments.reply_to_comment("user-1", "comment-1", "Hello"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(comments.update_comment("user-1", "comment-1", "Hello"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(comments.delete_comment("user-1", "comment-1"))
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(comments.moderate_comments("user-1", ["comment-1"], "rejected"))


def test_thread_filters_require_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        comments._build_thread_list_params(video_id="video-1", channel_id="channel-1")


def test_moderation_validation() -> None:
    with pytest.raises(ValueError, match="more than 50"):
        asyncio.run(comments.moderate_comments("user-1", ["comment"] * 51, "rejected", approved=True))
    with pytest.raises(ValueError, match="only valid"):
        asyncio.run(comments.moderate_comments("user-1", ["comment-1"], "published", ban_author=True, approved=True))


@pytest.mark.asyncio
async def test_list_threads_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse({"items": [{"id": "thread-1"}], "nextPageToken": "next"}))
    monkeypatch.setattr(comments, "_manager_token", fake_manager_token)
    monkeypatch.setattr(comments.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await comments.list_comment_threads(
        "user-1",
        video_id="video-1",
        max_results=10,
        page_token="page",
        order="time",
    )

    assert result["comments"] == [{"id": "thread-1"}]
    assert fake_client.calls[0][2]["params"] == {
        "part": "snippet,replies",
        "maxResults": 10,
        "textFormat": "plainText",
        "videoId": "video-1",
        "pageToken": "page",
        "order": "time",
    }


@pytest.mark.asyncio
async def test_reply_sends_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse({"id": "comment-2"}))
    monkeypatch.setattr(comments, "_manager_token", fake_manager_token)
    monkeypatch.setattr(comments.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await comments.reply_to_comment("user-1", "comment-1", "Thanks!", approved=True)

    assert result["comment"] == {"id": "comment-2"}
    call = fake_client.calls[0][2]
    assert call["params"] == {"part": "snippet"}
    assert call["json"] == {"snippet": {"parentId": "comment-1", "textOriginal": "Thanks!"}}


@pytest.mark.asyncio
async def test_delete_handles_empty_204_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_manager_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(FakeResponse(status_code=204))
    monkeypatch.setattr(comments, "_manager_token", fake_manager_token)
    monkeypatch.setattr(comments.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await comments.delete_comment("user-1", "comment-1", approved=True)

    assert result["deleted"] is True
    assert fake_client.calls[0][0] == "DELETE"
    assert fake_client.calls[0][2]["params"] == {"id": "comment-1"}
