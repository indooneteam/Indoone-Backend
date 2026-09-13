import base64
import hashlib
import hmac
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.ai.conversation_store import ConversationStore
from app.api.request_context import clear_principal_id, get_principal_id, require_principal_id, set_principal_id
from app.main import app


client = TestClient(app)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = _encode(user_id.encode("utf-8"))
    timestamp = _encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_message(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    response = client.post("/api/chat", json={"message": ""}, headers=_auth_headers())
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.headers["X-Request-ID"]


def test_chat_returns_ai_reply(monkeypatch) -> None:
    async def fake_reply(message: str, history=None, document_context="") -> str:
        assert history == []
        assert document_context == ""
        return f"test reply: {message}"

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)

    response = client.post("/api/chat", json={"message": "Hello Indoone"}, headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "test reply: Hello Indoone"
    assert payload["conversation_id"]
    assert response.headers["X-Request-ID"]


def test_chat_reuses_conversation_context(monkeypatch, tmp_path: Path) -> None:
    observed: list[list[tuple[str, str]]] = []

    async def fake_reply(message: str, history=None, document_context="") -> str:
        assert document_context == ""
        observed.append(history or [])
        return f"reply: {message}"

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)
    monkeypatch.setattr(
        "app.api.chat._store",
        ConversationStore(tmp_path / "api-conversations.sqlite3"),
    )

    headers = _auth_headers()
    first = client.post("/api/chat", json={"message": "Hello"}, headers=headers)
    conversation_id = first.json()["conversation_id"]
    second = client.post(
        "/api/chat",
        json={"message": "What did I say?", "conversation_id": conversation_id},
        headers=headers,
    )

    assert second.status_code == 200
    assert observed[0] == []
    assert observed[1] == [("user", "Hello"), ("assistant", "reply: Hello")]


def test_principal_context_can_be_cleared() -> None:
    set_principal_id("user-one")
    assert require_principal_id() == "user-one"
    clear_principal_id()
    assert get_principal_id() == ""
