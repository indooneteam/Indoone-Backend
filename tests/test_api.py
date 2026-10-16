import base64
import hashlib
import hmac
import logging
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


def test_health_emits_structured_request_log(caplog) -> None:
    caplog.set_level(logging.INFO, logger="indoone.api")

    response = client.get("/health")

    assert response.status_code == 200
    records = [record for record in caplog.records if record.name == "indoone.api" and record.message == "request completed"]
    assert records
    record = records[-1]
    assert record.request_id == response.headers["X-Request-ID"]
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert isinstance(record.duration_ms, float)


def test_unhandled_exception_emits_structured_error_log(monkeypatch, caplog) -> None:
    def fail_health_details() -> dict[str, object]:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("app.main.sqlite_runtime_status", fail_health_details)
    caplog.set_level(logging.ERROR, logger="indoone.api")
    error_client = TestClient(app, raise_server_exceptions=False)

    response = error_client.get("/health/details")

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    records = [record for record in caplog.records if record.name == "indoone.api" and record.message == "unhandled request exception"]
    assert records
    record = records[-1]
    assert record.request_id == response.headers["X-Request-ID"]
    assert record.method == "GET"
    assert record.path == "/health/details"
    assert record.route == "/health/details"
    assert record.exception_type == "RuntimeError"


def test_health_details_exposes_runtime_metrics() -> None:
    client.get("/health")

    response = client.get("/health/details")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["uptime_seconds"], (int, float))
    assert payload["uptime_seconds"] >= 0
    assert payload["requests_total"] >= 1
    assert payload["responses_by_status"]["200"] >= 1
    assert payload["sqlite"]["foreign_keys"] is True


def test_ready_reports_readiness() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert response.headers["X-Request-ID"]


def test_ready_returns_503_when_sqlite_unavailable(monkeypatch, caplog) -> None:
    def fail_readiness() -> None:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("app.main.sqlite_runtime_status", fail_readiness)
    caplog.set_level(logging.ERROR, logger="indoone.api")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["code"] == "NOT_READY"
    assert response.headers["X-Request-ID"]
    assert any(record.message == "readiness check failed" for record in caplog.records)


def test_chat_rejects_empty_message(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    response = client.post("/api/chat", json={"message": ""}, headers=_auth_headers())
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.headers["X-Request-ID"]


def test_chat_returns_ai_reply(monkeypatch) -> None:
    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        assert history == []
        assert document_context == ""
        return f"test reply: {message}"

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)

    response = client.post("/api/chat", json={"message": "Hello Indoone"}, headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "test reply: Hello Indoone"
    assert payload["sources"] == []
    assert payload["conversation_id"]
    assert response.headers["X-Request-ID"]


def test_chat_does_not_leak_the_provider_error_to_the_client(monkeypatch, caplog) -> None:
    """A provider failure must reach the client as our wording, never as the provider's.

    The Gemini service raises a RuntimeError whose text carries the provider's own error
    body. The API must log that text and answer with wording of our own, because the app
    shows this detail to the user as the AI's reply.
    """

    async def failing_reply(
        message: str, history=None, document_context="", extra_system_instruction=""
    ) -> str:
        raise RuntimeError(
            "Gemini API returned HTTP 400: API key not valid. Please pass a valid API key."
        )

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setattr("app.api.chat.generate_reply", failing_reply)

    with caplog.at_level(logging.ERROR, logger="indoone.chat"):
        response = client.post(
            "/api/chat",
            json={"message": "Hello Indoone"},
            headers=_auth_headers(),
        )

    assert response.status_code == 503
    # The HTTPException handler puts exc.detail on the response as "message", and that is
    # the field the app reads first.
    payload = response.json()
    assert payload["code"] == "HTTP_503"
    message = payload["message"]
    assert message == (
        "Indoone AI could not complete this request. Please try again in a moment."
    )
    assert "Gemini" not in message
    assert "API key" not in message
    # The real reason is not lost: it is in the log, where it is useful.
    assert "Gemini API returned HTTP 400" in caplog.text


def test_chat_reuses_conversation_context(monkeypatch, tmp_path: Path) -> None:
    observed: list[list[tuple[str, str]]] = []

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
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


def test_chat_returns_grounded_research_sources(monkeypatch) -> None:
    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return (
            "AI technology is developing across several research areas.\n\n"
            "Sources:\n"
            "1. Research source — https://research.example/ai\n"
            "2. News source — https://news.example/ai"
        )

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setattr("app.api.chat.generate_reply", fake_reply)

    response = client.post(
        "/api/chat",
        json={"message": "Research AI technology"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {"title": "Research source", "url": "https://research.example/ai"},
        {"title": "News source", "url": "https://news.example/ai"},
    ]


def test_user_name_line_uses_the_stored_name_only(monkeypatch) -> None:
    """Only the name reaches the prompt; nothing else stored about the user does."""
    from app.api import chat

    monkeypatch.setattr(
        chat,
        "list_memories",
        lambda user_id, limit=20: [
            {"key": "profession", "value": "Student"},
            {"key": "name", "value": "  Yashwanth  "},
        ],
    )
    assert chat._user_name_line("u1") == "The user's name is Yashwanth."

    monkeypatch.setattr(chat, "list_memories", lambda user_id, limit=20: [])
    assert chat._user_name_line("u1") == ""

    monkeypatch.setattr(chat, "list_memories", lambda user_id, limit=20: [{"key": "nickname", "value": "Chinni"}])
    assert chat._user_name_line("u1") == "The user's name is Chinni."

    assert chat._user_name_line("") == ""


def test_now_line_gives_today_in_ist() -> None:
    """The model must be told today's date, or it answers from training memory."""
    from app.api import chat

    line = chat._now_line()
    assert line.startswith("Today is ")
    assert line.endswith(" IST.")
    today = chat.datetime.now(chat._IST).strftime("%Y-%m-%d")
    assert today in line
