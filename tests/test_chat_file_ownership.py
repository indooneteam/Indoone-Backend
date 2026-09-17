import pytest
from fastapi import Request

from app.api import chat as chat_api


def test_chat_reads_file_with_authenticated_owner(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_read(file_id: str, user_id: str = "") -> str:
        captured["file_id"] = file_id
        captured["user_id"] = user_id
        return "owned content"

    monkeypatch.setattr(chat_api, "read_text_file", fake_read)
    request = Request({"type": "http", "headers": [], "method": "POST", "path": "/api/chat", "query_string": b"", "server": ("test", 80), "scheme": "http", "client": ("test", 1)})
    request.state.principal_id = "user-one"

    assert chat_api._principal(request) == "user-one"
    assert fake_read("file-1", user_id=chat_api._principal(request)) == "owned content"
    assert captured == {"file_id": "file-1", "user_id": "user-one"}
