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
    async def fake_reply(message: str) -> str:
        return f"test reply: {message}"

    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)

    response = client.post("/api/chat", json={"message": "Hello Indoone"})

    assert response.status_code == 200
    assert response.json() == {"reply": "test reply: Hello Indoone"}
