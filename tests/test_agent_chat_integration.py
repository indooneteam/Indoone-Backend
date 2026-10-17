import base64
import hashlib
import hmac
import time

from fastapi.testclient import TestClient

import pytest

from app.ai.agent import AgentExecution, AgentStep
from app.ai.tools import ToolResult
from app.main import app


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
    timestamp = base64.urlsafe_b64encode(str(int(time.time())).encode("ascii")).decode("ascii").rstrip("=")
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = base64.urlsafe_b64encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest()).decode("ascii").rstrip("=")
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


def test_chat_uses_model_for_tool_requests(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)

    async def fake_execute_agent_async(message: str, **kwargs) -> AgentExecution:
        assert message == "text stats: hello"
        step = AgentStep(index=1, tool="text_stats", payload="hello")
        return AgentExecution(
            message=message,
            steps=(step,),
            results=(ToolResult("text_stats", "words: 1"),),
        )

    captured_history: list[tuple[str, str]] = []

    async def fake_generate_reply(message: str, history=None, document_context="", extra_system_instruction: str = "") -> str:
        captured_history.extend(history or [])
        return "The trained model summarized the tool result."

    monkeypatch.setattr("app.api.chat.execute_agent_async", fake_execute_agent_async)
    monkeypatch.setattr("app.api.chat.generate_reply", fake_generate_reply)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "text stats: hello"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "The trained model summarized the tool result."
    assert any(
        role == "tool" and "words: 1" in content
        for role, content in captured_history
    )


def test_chat_does_not_hand_an_email_request_to_the_agent(monkeypatch, tmp_path) -> None:
    """An email request keeps its own path; the agent must not take it over.

    The agent plans gmail_compose for a compose request, and that tool needs no approval
    and no connector, so it used to run, set agent_ran, and skip the email flow entirely.
    The user then confirmed into a path that never ran, and nothing was sent.
    """
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    called = {"agent": False}

    async def fake_execute_agent_async(message: str, **kwargs) -> AgentExecution:
        called["agent"] = True
        return AgentExecution(message=message, steps=(), results=())

    async def fake_email_handler(user_id, conversation_id, message, history=None):
        return "The email path handled this."

    monkeypatch.setattr("app.api.chat.execute_agent_async", fake_execute_agent_async)
    monkeypatch.setattr("app.api.chat.handle_email_chat_with_tools", fake_email_handler)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "send an email to ravi about the meeting"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert called["agent"] is False, "the agent ran for an email request"
    assert response.json()["reply"] == "The email path handled this."
