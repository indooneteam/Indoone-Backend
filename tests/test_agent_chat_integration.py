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


def test_an_email_request_reaches_the_email_handler(monkeypatch, tmp_path) -> None:
    """An email request is answered by the email handler, whatever the agent plans.

    The agent still runs - it decides. What it must not do is execute an email step of
    its own: every email tool it could plan is either gated on the user having connected
    that mailbox or gated on approval, so it produces no result and leaves the request to
    the email path, which knows the provider and asks the one confirmation.
    """
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    handled: list[str] = []

    async def fake_email_handler(user_id, conversation_id, message, history=None):
        handled.append(message)
        return "The email path handled this."

    monkeypatch.setattr("app.api.chat.handle_email_chat_with_tools", fake_email_handler)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "send an email to ravi about the meeting"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert handled == ["send an email to ravi about the meeting"]
    assert response.json()["reply"] == "The email path handled this."


def test_the_agent_blocks_an_email_step_instead_of_running_it() -> None:
    """A compose step must come back blocked, so the chat never sends on its own.

    Two gates do this. The connector gate stops it for a user who has not connected that
    mailbox, and the approval gate stops it for a user who has. This checks the connector
    gate, which is the one that applies to a Zoho user.
    """
    import asyncio

    from app.ai.agent_async import execute_agent_async

    execution = asyncio.run(execute_agent_async("compose an email to ravi", user_id="user-1"))

    assert execution.results == (), "the agent executed an email step on its own"
    assert [step.tool for step in execution.blocked_steps] == ["gmail_compose"]


def test_a_compose_step_is_blocked_even_when_the_mailbox_is_connected(monkeypatch, tmp_path) -> None:
    """The approval gate, not just the connector gate.

    A user who has connected the mailbox passes the connector check, so without an
    approval requirement the agent would run gmail_compose and take the request away from
    the email path - which is exactly the regression this pins. With the requirement the
    step comes back blocked, the agent produces no result, and the email path keeps it.
    """
    import asyncio

    from app.ai.agent_async import execute_agent_async
    from app.capabilities.store import upsert_integration_token

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token(
        "user-1",
        "gmail",
        b"encrypted",
        None,
        "Bearer",
        "gmail.readonly",
        None,
    )

    execution = asyncio.run(execute_agent_async("compose an email to ravi", user_id="user-1"))

    assert execution.results == (), "the agent executed an email step on its own"
    assert [step.tool for step in execution.blocked_steps] == ["gmail_compose"]
