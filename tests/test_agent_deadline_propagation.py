from __future__ import annotations

import time

from app.ai.agent import execute_agent
from app.ai.tools import ToolResult


def test_agent_propagates_remaining_runtime_to_tool(monkeypatch) -> None:
    observed: list[float] = []

    def bounded_tool(name: str, payload: str, max_runtime_seconds: float | None = None) -> ToolResult:
        assert max_runtime_seconds is not None
        observed.append(max_runtime_seconds)
        time.sleep(0.005)
        return ToolResult(name, payload, safe=True, retryable=False)

    monkeypatch.setattr("app.ai.agent.run_tool", bounded_tool)
    monkeypatch.setattr("app.ai.agent.MAX_AGENT_RUNTIME_SECONDS", 0.05)

    execution = execute_agent("text stats: one; text stats: two")

    assert execution.results
    assert observed
    assert 0 < observed[0] <= 0.05


def test_agent_does_not_start_retry_after_deadline(monkeypatch) -> None:
    calls = 0

    def retryable_tool(name: str, payload: str, max_runtime_seconds: float | None = None) -> ToolResult:
        nonlocal calls
        calls += 1
        assert max_runtime_seconds is not None
        return ToolResult(name, "temporary", safe=False, retryable=True)

    monkeypatch.setattr("app.ai.agent.run_tool", retryable_tool)
    monkeypatch.setattr("app.ai.agent.MAX_AGENT_RUNTIME_SECONDS", 0.01)
    monkeypatch.setattr("app.ai.agent.time.monotonic", lambda: 100.0)

    execution = execute_agent("text stats: one")

    assert calls == 1
    assert execution.retry_counts == (0,)
    assert execution.results[0].safe is False
