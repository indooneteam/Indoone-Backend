from fastapi.testclient import TestClient

from app.main import app


def test_agent_api_exposes_result_execution_metadata() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "what is 9 + 4"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["retryable"] is False
    assert body["result"]["truncated"] is False
    assert body["results"][0]["retryable"] is False
    assert body["results"][0]["truncated"] is False


def test_agent_api_exposes_truncated_result_metadata(monkeypatch) -> None:
    from app.ai.tools import ToolResult

    async def fake_execute_agent_async(*args, **kwargs):
        from app.ai.agent import AgentExecution, AgentStep

        step = AgentStep(index=1, tool="text_stats", payload="hello")
        result = ToolResult("text_stats", "partial", safe=True, truncated=True)
        return AgentExecution(message="text stats: hello", steps=(step,), results=(result,))

    # /api/agent is served by the async agent router, which is registered
    # before the legacy capabilities router.
    monkeypatch.setattr("app.api.agent_async.execute_agent_async", fake_execute_agent_async)

    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "text stats: hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["truncated"] is True
    assert body["results"][0]["truncated"] is True
