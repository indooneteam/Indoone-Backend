import asyncio

import pytest

from app.capabilities import instagram_comments


def test_validate_id_rejects_invalid_values() -> None:
    assert instagram_comments._validate_id("media-1", "media_id") == "media-1"
    with pytest.raises(ValueError):
        instagram_comments._validate_id("bad id", "media_id")
    with pytest.raises(ValueError):
        instagram_comments._validate_id("", "comment_id")


def test_validate_message_bounds() -> None:
    assert instagram_comments._validate_message(" hello ") == "hello"
    with pytest.raises(ValueError):
        instagram_comments._validate_message("")
    with pytest.raises(ValueError):
        instagram_comments._validate_message("x" * 2201)


def test_list_media_comments_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert user_id == "user-1"
        assert method == "GET"
        assert path == "media-1/comments"
        assert kwargs["params"] == {
            "fields": "id,text,username,timestamp,like_count,replies{id,text,username,timestamp}",
            "limit": 10,
            "after": "next-token",
        }
        return {"data": [{"id": "c1", "text": "hello"}], "paging": {"next": "x"}}

    monkeypatch.setattr(instagram_comments, "_request_json", fake_request_json)
    result = asyncio.run(instagram_comments.list_media_comments("user-1", "media-1", 10, "next-token"))

    assert result["media_id"] == "media-1"
    assert result["data"][0]["id"] == "c1"
    assert result["secrets_exposed"] is False


def test_writes_require_explicit_approval() -> None:
    with pytest.raises(PermissionError):
        asyncio.run(instagram_comments.reply_to_comment("u", "c1", "hello"))
    with pytest.raises(PermissionError):
        asyncio.run(instagram_comments.create_media_comment("u", "m1", "hello"))
    with pytest.raises(PermissionError):
        asyncio.run(instagram_comments.delete_comment("u", "c1"))
    with pytest.raises(PermissionError):
        asyncio.run(instagram_comments.set_comment_hidden("u", "c1", True))


def test_reply_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("u", "POST", "c1/replies")
        assert kwargs["data"] == {"message": "Thanks"}
        return {"id": "r1"}

    monkeypatch.setattr(instagram_comments, "_request_json", fake_request_json)
    result = asyncio.run(instagram_comments.reply_to_comment("u", "c1", " Thanks ", approved=True))

    assert result["replied"] is True
    assert result["result"]["id"] == "r1"


def test_moderation_builds_hide_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("u", "POST", "c1")
        assert kwargs["data"] == {"hide": "true"}
        return {}

    monkeypatch.setattr(instagram_comments, "_request_json", fake_request_json)
    result = asyncio.run(instagram_comments.set_comment_hidden("u", "c1", True, approved=True))

    assert result["hidden"] is True
    assert result["moderated"] is True
