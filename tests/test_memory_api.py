from types import SimpleNamespace

import pytest

import app.api.memory as memory_api


@pytest.mark.asyncio
async def test_memory_write_uses_authenticated_principal(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(memory_api, "current_user_id", lambda request: "user-1")

    def fake_write_candidate(user_id: str, **kwargs):
        captured["user_id"] = user_id
        captured.update(kwargs)
        return {"id": "memory-1", "user_id": user_id, "key": kwargs["key"], "value": kwargs["value"]}

    monkeypatch.setattr(memory_api._memory_service, "write_candidate", fake_write_candidate)

    request = SimpleNamespace()
    body = memory_api.MemoryWriteRequest(key="name", value="Ravi", confidence=0.99)
    result = await memory_api.write_memory(request, body)

    assert result["memory"]["user_id"] == "user-1"
    assert captured["user_id"] == "user-1"
    assert captured["explicit_user_request"] is True


@pytest.mark.asyncio
async def test_memory_read_and_delete_are_user_scoped(monkeypatch) -> None:
    monkeypatch.setattr(memory_api, "current_user_id", lambda request: "user-2")

    captured: dict[str, object] = {}

    class FakeService:
        def search(self, user_id: str, query: str, limit: int = 10):
            captured["search_user_id"] = user_id
            return [{"id": "m2", "user_id": user_id, "key": "name", "value": "Ravi"}]

        def delete(self, user_id: str, memory_id: str):
            captured["delete_user_id"] = user_id
            captured["delete_memory_id"] = memory_id
            return True

    monkeypatch.setattr(memory_api, "_memory_service", FakeService())

    request = SimpleNamespace()
    result = await memory_api.get_memories(request, q="name", limit=5)
    deleted = await memory_api.remove_memory(request, memory_api.MemoryDeleteRequest(memory_id="m2"))

    assert result["memories"][0]["user_id"] == "user-2"
    assert captured["search_user_id"] == "user-2"
    assert captured["delete_user_id"] == "user-2"
    assert captured["delete_memory_id"] == "m2"


def test_memory_write_request_has_no_client_user_id_field() -> None:
    body = memory_api.MemoryWriteRequest(key="name", value="Ravi")
    assert not hasattr(body, "user_id")
