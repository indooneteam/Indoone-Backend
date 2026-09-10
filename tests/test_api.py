from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_message() -> None:
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_returns_ai_reply(monkeypatch) -> None:
    async def fake_reply(message: str, history=None) -> str:
        assert history == []
        return f"test reply: {message}"

    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)

    response = client.post("/api/chat", json={"message": "Hello Indoone"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "test reply: Hello Indoone"
    assert payload["conversation_id"]


def test_chat_reuses_conversation_context(monkeypatch) -> None:
    observed: list[list[tuple[str, str]]] = []

    async def fake_reply(message: str, history=None) -> str:
        observed.append(history or [])
        return f"reply: {message}"

    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)
    monkeypatch.setattr("app.api.chat._store", __import__("app.ai.conversation_store", fromlist=["ConversationStore"]).ConversationStore(":memory:"))

    first = client.post("/api/chat", json={"message": "Hello"})
    conversation_id = first.json()["conversation_id"]
    second = client.post(
        "/api/chat",
        json={"message": "What did I say?", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert observed[0] == []
    assert observed[1] == [("user", "Hello"), ("assistant", "reply: Hello")]
