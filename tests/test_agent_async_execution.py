from __future__ import annotations

import asyncio

from app.ai.agent_async import execute_agent_async
from app.ai.tools import ToolResult


def test_async_agent_uses_current_event_loop(monkeypatch) -> None:
    async def fake_run_tool_async(name: str, payload: str, max_runtime_seconds: float | None = None) -> ToolResult:
        return ToolResult(name, str(asyncio.get_running_loop()), safe=True)

    monkeypatch.setattr("app.ai.agent_async.run_tool_async", fake_run_tool_async)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        execution = await execute_agent_async("text stats: hello")
        assert execution.steps[0].tool == "text_stats"
        assert execution.results[0].output == str(loop)
        assert execution.results[0].safe is True

    asyncio.run(scenario())


def test_async_agent_retries_retryable_result(monkeypatch) -> None:
    calls = 0

    async def fake_run_tool_async(name: str, payload: str, max_runtime_seconds: float | None = None) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ToolResult(name, "temporary", safe=False, retryable=True)
        return ToolResult(name, "recovered", safe=True)

    monkeypatch.setattr("app.ai.agent_async.run_tool_async", fake_run_tool_async)

    async def scenario() -> None:
        execution = await execute_agent_async("text stats: hello")
        assert execution.results[0].output == "recovered"
        assert execution.retry_counts == (1,)
        assert calls == 2

    asyncio.run(scenario())
