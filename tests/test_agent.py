from __future__ import annotations

from fastapi.testclient import TestClient

from app.ai.agent import ALLOWED_AGENT_TOOLS, MAX_AGENT_STEPS, build_agent_steps, execute_agent
from app.main import app


def test_agent_builds_calculator_step() -> None:
    steps = build_agent_steps("what is 12 + 30")
    assert len(steps) == 1
    assert steps[0].tool == "calculator"
    assert steps[0].payload == "12+30"


def test_agent_builds_multiple_bounded_steps() -> None:
    steps = build_agent_steps("calculate 2 + 3, then 7 * 8, then 100 / 4")
    assert [step.payload for step in steps] == ["2+3", "7*8", "100/4"]
    assert [step.index for step in steps] == [1, 2, 3]
    assert all(step.tool in ALLOWED_AGENT_TOOLS for step in steps)


def test_agent_is_bounded_and_returns_tool_results() -> None:
    execution = execute_agent("calculate 7 * 8, 10 + 5, 100 / 4, 9 % 2, 1 + 1")
    assert len(execution.steps) == MAX_AGENT_STEPS
    assert [result.output for result in execution.results] == ["56", "15", "25.0", "1",]
    assert all(result.safe for result in execution.results)


def test_agent_endpoint_exposes_steps() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "what is 9 + 4"})
    assert response.status_code == 200
    body = response.json()
    assert body["max_steps"] == MAX_AGENT_STEPS
    assert body["steps"][0]["tool"] == "calculator"
    assert body["results"][0]["output"] == "13"


def test_agent_endpoint_exposes_multiple_results() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "9 + 4; 3 * 6"})
    assert response.status_code == 200
    body = response.json()
    assert [step["payload"] for step in body["steps"]] == ["9+4", "3*6"]
    assert [result["output"] for result in body["results"]] == ["13", "18"]


def test_agent_endpoint_returns_empty_plan_for_general_message() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "hello Indoone"})
    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == []
    assert body["results"] == []
