import asyncio

import pytest

from app.capabilities import instagram_messaging


def test_validate_id_and_text() -> None:
    assert instagram_messaging._validate_id("recipient-1", "recipient_id") == "recipient-1"
    assert instagram_messaging._validate_text(" hello ") == "hello"
    with pytest.raises(ValueError):
        instagram_messaging._validate_id("bad id", "recipient_id")
    with pytest.raises(ValueError):
        instagram_messaging._validate_text("")
    with pytest.raises(ValueError):
        instagram_messaging._validate_text("x" * 1001)


def test_list_conversations_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("user-1", "GET", "me/conversations")
        assert kwargs["params"] == {
            "platform": "instagram",
            "limit": 10,
            "after": "cursor-1",
            "user_id": "target-1",
        }
        return {"data": [{"id": "conv-1"}], "paging": {"next": "cursor-2"}}

    monkeypatch.setattr(instagram_messaging, "_request_json", fake_request_json)
    result = asyncio.run(
        instagram_messaging.list_conversations("user-1", 10, "cursor-1", "target-1")
    )
    assert result["data"][0]["id"] == "conv-1"
    assert result["secrets_exposed"] is False


def test_conversation_messages_and_message_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        if path == "conv-1":
            assert kwargs["params"] == {"fields": "messages"}
            return {"messages": {"data": [{"id": "msg-1"}]}}
        assert path == "msg-1"
        assert kwargs["params"] == {"fields": "id,created_time,from,to,message"}
        return {"id": "msg-1", "message": "Hello"}

    monkeypatch.setattr(instagram_messaging, "_request_json", fake_request_json)
    conversation = asyncio.run(instagram_messaging.list_conversation_messages("u", "conv-1"))
    message = asyncio.run(instagram_messaging.get_message("u", "msg-1"))
    assert conversation["messages"]["data"][0]["id"] == "msg-1"
    assert message["message"]["message"] == "Hello"


def test_sends_require_explicit_approval() -> None:
    with pytest.raises(PermissionError):
        asyncio.run(instagram_messaging.send_text_message("u", "target", "hello"))
    with pytest.raises(PermissionError):
        asyncio.run(instagram_messaging.send_media_share("u", "target", "media-1"))


def test_send_text_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("u", "POST", "me/messages")
        assert kwargs["json"] == {
            "recipient": {"id": "target"},
            "message": {"text": "Hello"},
        }
        return {"recipient_id": "target", "message_id": "msg-1"}

    monkeypatch.setattr(instagram_messaging, "_request_json", fake_request_json)
    result = asyncio.run(instagram_messaging.send_text_message("u", "target", " Hello ", True))
    assert result["sent"] is True
    assert result["result"]["message_id"] == "msg-1"


def test_send_media_share_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert kwargs["json"] == {
            "recipient": {"id": "target"},
            "message": {"attachment": {"type": "MEDIA_SHARE", "payload": {"id": "media-1"}}},
        }
        return {"message_id": "msg-2"}

    monkeypatch.setattr(instagram_messaging, "_request_json", fake_request_json)
    result = asyncio.run(instagram_messaging.send_media_share("u", "target", "media-1", True))
    assert result["media_id"] == "media-1"
    assert result["result"]["message_id"] == "msg-2"
