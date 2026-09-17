import asyncio

import pytest

from app.capabilities import instagram_publishing


def test_carousel_validation() -> None:
    assert instagram_publishing._validate_image_urls([" a ", "b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        instagram_publishing._validate_image_urls(["one"])
    with pytest.raises(ValueError):
        instagram_publishing._validate_image_urls(["x"] * 11)
    with pytest.raises(ValueError):
        instagram_publishing._validate_image_urls(["a", ""])


def test_writes_require_approval() -> None:
    with pytest.raises(PermissionError):
        asyncio.run(instagram_publishing.create_carousel_container("u", ["a", "b"]))
    with pytest.raises(PermissionError):
        asyncio.run(instagram_publishing.publish_ready_container("u", "c1"))


def test_publish_requires_finished_container(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        calls.append((method, path))
        return {"id": "c1", "status": "IN_PROGRESS"}

    monkeypatch.setattr(instagram_publishing, "_request_json", fake_request_json)

    with pytest.raises(ValueError, match="not ready"):
        asyncio.run(instagram_publishing.publish_ready_container("u", "c1", approved=True))
    assert calls == [("GET", "c1")]


def test_publish_finished_container(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        calls.append((method, path, kwargs.get("data")))
        if method == "GET":
            return {"id": "c1", "status": "FINISHED", "status_code": "FINISHED"}
        return {"id": "published-1"}

    async def fake_access_token(user_id: str) -> str:
        return "token"

    monkeypatch.setattr(instagram_publishing, "_request_json", fake_request_json)
    monkeypatch.setattr(instagram_publishing, "_access_token", fake_access_token)

    result = asyncio.run(instagram_publishing.publish_ready_container("u", "c1", approved=True))

    assert calls[0][0:2] == ("GET", "c1")
    assert calls[1][0:2] == ("POST", "me/media_publish")
    assert result["published"] is True
    assert result["container_status"] == "FINISHED"
