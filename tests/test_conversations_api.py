import base64
import hashlib
import hmac
import time

from fastapi.testclient import TestClient

from app.ai.conversation_store import ConversationStore
from app.api.conversations import _store
from app.main import app

client = TestClient(app)


def _auth_headers(user_id: str) -> dict[str, str]:
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    user = encode(user_id.encode())
    timestamp = encode(str(int(time.time())).encode())
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


def test_conversation_routes_require_auth(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    response = client.get("/api/conversations")
    assert response.status_code == 401


def test_conversation_list_close_delete_are_owner_scoped(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr("app.api.conversations._store", store)
    store.append("c1", [("user", "private")], user_id="user-one")

    mine = client.get("/api/conversations", headers=_auth_headers("user-one"))
    assert mine.status_code == 200
    assert mine.json()["conversations"][0]["conversation_id"] == "c1"

    blocked = client.post("/api/conversations/c1/close", headers=_auth_headers("user-two"))
    assert blocked.status_code == 403

    closed = client.post("/api/conversations/c1/close", headers=_auth_headers("user-one"))
    assert closed.status_code == 200
    assert closed.json() == {"conversation_id": "c1", "status": "closed"}

    deleted = client.delete("/api/conversations/c1", headers=_auth_headers("user-one"))
    assert deleted.status_code == 204
