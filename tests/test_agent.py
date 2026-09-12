from __future__ import annotations

from fastapi.testclient import TestClient

from app.ai.agent import MAX_AGENT_STEPS, build_agent_steps, execute_agent
from app.main import app


def test_agent_builds_calculator_step() -> None:
    steps = build_agent_steps("what is 12 + 30")
    assert len(steps) == 1
    assert steps[0].tool == "calculator"
    assert steps[0].payload == "12+30"


def test_agent_is_bounded_and_returns_tool_result() -> None:
    execution = execute_agent("calculate 7 * 8")
    assert len(execution.steps) <= MAX_AGENT_STEPS
    assert execution.results[0].output == "56"
    assert execution.results[0].safe is True


def test_agent_endpoint_exposes_steps() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "what is 9 + 4"})
    assert response.status_code == 200
    body = response.json()
    assert body["max_steps"] == MAX_AGENT_STEPS
    assert body["steps"][0]["tool"] == "calculator"
    assert body["results"][0]["output"] == "13"


def test_agent_endpoint_returns_empty_plan_for_general_message() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "hello Indoone"})
    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == []
    assert body["results"] == []
