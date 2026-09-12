from __future__ import annotations

import os
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app.ai.agent import ALLOWED_AGENT_TOOLS, MAX_AGENT_STEPS, build_agent_steps, execute_agent
from app.main import app
from app.capabilities.store import get_agent_run, upsert_memory


def test_agent_builds_calculator_step() -> None:
    steps = build_agent_steps("what is 12 + 30")
    assert len(steps) == 1
    assert steps[0].tool == "calculator"
    assert steps[0].payload == "12+30"
    assert steps[0].requires_approval is False


def test_agent_builds_multiple_bounded_steps() -> None:
    steps = build_agent_steps("calculate 2 + 3, then 7 * 8, then 100 / 4")
    assert [step.payload for step in steps] == ["2+3", "7*8", "100/4"]
    assert [step.index for step in steps] == [1, 2, 3]
    assert all(step.tool in ALLOWED_AGENT_TOOLS for step in steps)


def test_agent_routes_explicit_tools_in_message_order() -> None:
    steps = build_agent_steps("text stats: hello world; summarize json: {\"a\":1}")
    assert [step.tool for step in steps] == ["text_stats", "json_summary"]
    assert [step.index for step in steps] == [1, 2]


def test_agent_is_bounded_and_returns_tool_results() -> None:
    execution = execute_agent("calculate 7 * 8, 10 + 5, 100 / 4, 9 % 2, 1 + 1")
    assert len(execution.steps) == MAX_AGENT_STEPS
    assert [result.output for result in execution.results] == ["56", "15", "25.0", "1"]
    assert all(result.safe for result in execution.results)
    assert execution.retry_counts == (0, 0, 0, 0)


def test_agent_loads_memory_context() -> None:
    with TemporaryDirectory() as tempdir:
        old_path = os.environ.get("INDOONE_CAPABILITY_DB")
        os.environ["INDOONE_CAPABILITY_DB"] = os.path.join(tempdir, "agent.db")
        try:
            upsert_memory("user-1", "timezone", "Asia/Kolkata", 1.0, "user")
            execution = execute_agent("timezone", user_id="user-1")
            assert execution.memories
            assert execution.memories[0]["key"] == "timezone"
        finally:
            if old_path is None:
                os.environ.pop("INDOONE_CAPABILITY_DB", None)
            else:
                os.environ["INDOONE_CAPABILITY_DB"] = old_path


def test_agent_chains_previous_result_into_calculator() -> None:
    execution = execute_agent("calculate 2 + 3, then $last * 4")
    assert [result.output for result in execution.results] == ["5", "20"]


def test_agent_exposes_bounded_retry_state_for_failures() -> None:
    execution = execute_agent("calculate 10 / 0")
    assert len(execution.results) == 1
    assert execution.results[0].safe is False
    assert execution.retry_counts == (1,)


def test_agent_persists_execution_state() -> None:
    with TemporaryDirectory() as tempdir:
        old_path = os.environ.get("INDOONE_CAPABILITY_DB")
        os.environ["INDOONE_CAPABILITY_DB"] = os.path.join(tempdir, "agent.db")
        try:
            execution = execute_agent("calculate 2 + 3, then $last * 4", user_id="user-1")
            assert execution.run_id
            saved = get_agent_run("user-1", execution.run_id)
            assert saved is not None
            assert saved["status"] == "completed"
            assert [item["output"] for item in saved["results"]] == ["5", "20"]
            assert saved["retry_counts"] == [0, 0]
        finally:
            if old_path is None:
                os.environ.pop("INDOONE_CAPABILITY_DB", None)
            else:
                os.environ["INDOONE_CAPABILITY_DB"] = old_path


def test_agent_endpoint_exposes_steps() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "what is 9 + 4"})
    assert response.status_code == 200
    body = response.json()
    assert body["max_steps"] == MAX_AGENT_STEPS
    assert body["steps"][0]["tool"] == "calculator"
    assert body["results"][0]["output"] == "13"
    assert body["blocked_steps"] == []
    assert body["retry_counts"] == [0]
    assert body["run_id"] is None


def test_agent_endpoint_exposes_persisted_run_history() -> None:
    with TemporaryDirectory() as tempdir:
        old_path = os.environ.get("INDOONE_CAPABILITY_DB")
        os.environ["INDOONE_CAPABILITY_DB"] = os.path.join(tempdir, "agent.db")
        try:
            with TestClient(app) as client:
                response = client.post("/api/agent", json={"message": "9 + 4", "user_id": "user-1"})
                assert response.status_code == 200
                run_id = response.json()["run_id"]
                assert run_id

                detail = client.get(f"/api/agent/runs/{run_id}", params={"user_id": "user-1"})
                assert detail.status_code == 200
                assert detail.json()["run"]["status"] == "completed"

                history = client.get("/api/agent/runs", params={"user_id": "user-1"})
                assert history.status_code == 200
                runs = history.json()["runs"]
                assert len(runs) == 1
                assert runs[0]["id"] == run_id
        finally:
            if old_path is None:
                os.environ.pop("INDOONE_CAPABILITY_DB", None)
            else:
                os.environ["INDOONE_CAPABILITY_DB"] = old_path


def test_agent_run_is_user_scoped() -> None:
    with TemporaryDirectory() as tempdir:
        old_path = os.environ.get("INDOONE_CAPABILITY_DB")
        os.environ["INDOONE_CAPABILITY_DB"] = os.path.join(tempdir, "agent.db")
        try:
            execution = execute_agent("1 + 1", user_id="user-1")
            assert execution.run_id
            with TestClient(app) as client:
                response = client.get(f"/api/agent/runs/{execution.run_id}", params={"user_id": "user-2"})
            assert response.status_code == 404
        finally:
            if old_path is None:
                os.environ.pop("INDOONE_CAPABILITY_DB", None)
            else:
                os.environ["INDOONE_CAPABILITY_DB"] = old_path


def test_agent_endpoint_exposes_multiple_results() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "9 + 4; 3 * 6"})
    assert response.status_code == 200
    body = response.json()
    assert [step["payload"] for step in body["steps"]] == ["9+4", "3*6"]
    assert [result["output"] for result in body["results"]] == ["13", "18"]
    assert body["retry_counts"] == [0, 0]


def test_agent_endpoint_returns_empty_plan_for_general_message() -> None:
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"message": "hello Indoone"})
    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == []
    assert body["results"] == []
    assert body["blocked_steps"] == []
    assert body["retry_counts"] == []
