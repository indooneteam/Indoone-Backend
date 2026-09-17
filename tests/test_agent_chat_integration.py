import base64
import hashlib
import hmac
import time

from fastapi.testclient import TestClient

from app.ai.agent import AgentExecution, AgentStep
from app.ai.tools import ToolResult
from app.main import app


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
    timestamp = base64.urlsafe_b64encode(str(int(time.time())).encode("ascii")).decode("ascii").rstrip("=")
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = base64.urlsafe_b64encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest()).decode("ascii").rstrip("=")
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


def test_chat_uses_async_agent_for_tool_requests(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)

    async def fake_execute_agent_async(message: str, **kwargs) -> AgentExecution:
        assert message == "text stats: hello"
        step = AgentStep(index=1, tool="text_stats", payload="hello")
        return AgentExecution(
            message=message,
            steps=(step,),
            results=(ToolResult("text_stats", "words: 1"),),
        )

    monkeypatch.setattr("app.api.chat.execute_agent_async", fake_execute_agent_async)
    monkeypatch.setattr("app.api.chat.assess_answer", lambda *_: type("Q", (), {"passed": True})())

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "text stats: hello"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "words: 1"
