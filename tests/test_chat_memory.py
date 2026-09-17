from types import SimpleNamespace

import pytest

import app.api.chat as chat_api


@pytest.mark.asyncio
async def test_chat_includes_scoped_memory_in_model_history(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(chat_api, "current_user_id", lambda request: "user-1")

    class FakeMemoryService:
        def search(self, user_id: str, query: str, limit: int = 10):
            assert user_id == "user-1"
            assert query == "What should I call myself?"
            assert limit == 5
            return [{"key": "name", "value": "Ravi", "confidence": 0.99}]

    monkeypatch.setattr(chat_api, "_memory_service", FakeMemoryService())

    async def fake_agent(message: str, user_id: str):
        return SimpleNamespace(steps=[], blocked_steps=[], results=[])

    async def fake_generate(message: str, history=None, document_context=""):
        captured["history"] = history
        return "You said your name is Ravi."

    monkeypatch.setattr(chat_api, "execute_agent_async", fake_agent)
    monkeypatch.setattr(chat_api, "generate_reply", fake_generate)
    monkeypatch.setattr(chat_api._store, "is_closed", lambda conversation_id, user_id=None: False)
    monkeypatch.setattr(chat_api._store, "recent", lambda conversation_id, user_id=None: [])
    monkeypatch.setattr(chat_api._store, "append", lambda *args, **kwargs: None)

    request = SimpleNamespace(state=SimpleNamespace(principal_id="user-1"))
    response = await chat_api.chat(request, chat_api.ChatRequest(message="What should I call myself?"))

    assert response.reply == "You said your name is Ravi."
    assert captured["history"][-1] == ("memory", "User memory:\n- name: Ravi")
