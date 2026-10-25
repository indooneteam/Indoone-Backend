"""The async agent must pass the user id on a retry, exactly as the sync path does.

The retry call used to drop it, so a connector tool that hit a retryable timeout
was refused by the authorization layer ("authenticated user is required") instead
of being retried.
"""
from __future__ import annotations

import asyncio
import time

from app.ai.agents import agent_async
from app.ai.execution.tools import ToolResult


def test_the_retry_keeps_the_user_id(monkeypatch):
    calls: list[str | None] = []

    async def fake_run_tool_async(tool, payload, **kwargs):
        calls.append(kwargs.get("user_id"))
        if len(calls) == 1:
            return ToolResult(tool, "Tool error: timed out", safe=False, retryable=True)
        return ToolResult(tool, "ok", safe=True)

    monkeypatch.setattr(agent_async, "run_tool_async", fake_run_tool_async)

    result, retries = asyncio.run(
        agent_async._run_with_retry_async("calculator", "1+1", time.monotonic() + 5, user_id="user-9")
    )

    assert retries == 1
    assert result.safe is True
    assert calls == ["user-9", "user-9"], "the user id must travel on the retry too"
