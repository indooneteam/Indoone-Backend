from __future__ import annotations

import asyncio

from fastapi import Request

from app.api import chat as chat_api
from app.api.chat import ChatRequest


def test_chat_returns_grounded_sources_and_persists_full_reply(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStore:
        def is_closed(self, conversation_id: str, user_id: str = "") -> bool:
            return False

        def recent(self, conversation_id: str, user_id: str = "") -> list[tuple[str, str]]:
            return []

        def append(self, conversation_id, messages, user_id: str = "") -> None:
            captured["conversation_id"] = conversation_id
            captured["messages"] = messages
            captured["user_id"] = user_id

    async def fake_generate_reply(message, *, history=None, document_context=""):
        assert message == "What is the latest update?"
        return (
            "Current answer.\n\nSources:\n"
            "1. Official update — https://example.com/update"
        )

    monkeypatch.setattr(chat_api, "_store", FakeStore())
    monkeypatch.setattr(chat_api, "current_user_id", lambda request: "user-one")
    monkeypatch.setattr(chat_api, "generate_reply", fake_generate_reply)

    request = Request(
        {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/api/chat",
            "query_string": b"",
            "server": ("test", 80),
            "scheme": "http",
            "client": ("test", 1),
        }
    )
    response = asyncio.run(
        chat_api.chat(
            request,
            ChatRequest(message="What is the latest update?"),
        )
    )

    assert response.reply == "Current answer."
    assert [(item.title, item.url) for item in response.sources] == [
        ("Official update", "https://example.com/update")
    ]
    stored = captured["messages"]
    assert stored[1][1].endswith("1. Official update — https://example.com/update")
    assert captured["user_id"] == "user-one"
