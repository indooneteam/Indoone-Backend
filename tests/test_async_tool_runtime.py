from __future__ import annotations

import asyncio

from app.ai.tools import TOOLS, ToolResult, run_tool_async



def test_async_tool_runtime_keeps_async_tool_on_current_loop(monkeypatch) -> None:
    async def async_tool(_: str) -> str:
        return str(asyncio.get_running_loop())

    monkeypatch.setitem(TOOLS, "text_stats", async_tool)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        result = await run_tool_async("text_stats", "hello")
        assert result.safe is True
        assert result.output == str(loop)

    asyncio.run(scenario())



def test_async_tool_runtime_cancels_timed_out_coroutine(monkeypatch) -> None:
    cancelled = False

    async def slow_tool(_: str) -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(1)
            return "done"
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setitem(TOOLS, "text_stats", slow_tool)
    monkeypatch.setattr("app.ai.tools.get_tool_spec", lambda _: type("Spec", (), {"timeout_seconds": 0.01, "max_output_chars": 100})())

    async def scenario() -> ToolResult:
        return await run_tool_async("text_stats", "hello", max_runtime_seconds=0.01)

    result = asyncio.run(scenario())
    assert result.safe is False
    assert result.retryable is True
    assert "timed out" in result.output.lower()
    assert cancelled is True
