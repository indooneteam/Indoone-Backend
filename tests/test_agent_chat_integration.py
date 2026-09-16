from fastapi.testclient import TestClient

from app.ai.agent import AgentExecution, AgentStep
from app.ai.tools import ToolResult
from app.main import app


def test_chat_uses_async_agent_for_tool_requests(monkeypatch) -> None:
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
        response = client.post("/api/chat", json={"message": "text stats: hello"})

    assert response.status_code == 401
