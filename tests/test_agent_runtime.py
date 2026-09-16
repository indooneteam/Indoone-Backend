from __future__ import annotations

import time

from app.ai.agent import MAX_AGENT_RUNTIME_SECONDS, execute_agent
from app.ai.tools import ToolResult


def test_agent_stops_when_total_runtime_budget_is_exhausted(monkeypatch) -> None:
    calls = {"count": 0}
    started = time.monotonic()

    def slow_tool(name: str, payload: str) -> ToolResult:
        calls["count"] += 1
        time.sleep(0.02)
        return ToolResult(name, payload, safe=True, retryable=False)

    monkeypatch.setattr("app.ai.agent.run_tool", slow_tool)
    monkeypatch.setattr("app.ai.agent.MAX_AGENT_RUNTIME_SECONDS", 0.03)

    execution = execute_agent("text stats: one; text stats: two; text stats: three")
    elapsed = time.monotonic() - started

    assert calls["count"] == 2
    assert len(execution.results) == 2
    assert len(execution.blocked_steps) == 1
    assert execution.blocked_steps[0].index == 3
    assert elapsed < MAX_AGENT_RUNTIME_SECONDS


def test_agent_runtime_budget_is_independent_of_step_limit(monkeypatch) -> None:
    monkeypatch.setattr("app.ai.agent.MAX_AGENT_RUNTIME_SECONDS", 0.02)

    def slow_tool(name: str, payload: str) -> ToolResult:
        time.sleep(0.03)
        return ToolResult(name, payload)

    monkeypatch.setattr("app.ai.agent.run_tool", slow_tool)

    execution = execute_agent("text stats: one; text stats: two")

    assert len(execution.steps) == 1
    assert len(execution.results) == 1
    assert len(execution.blocked_steps) == 1
    assert execution.blocked_steps[0].index == 2
