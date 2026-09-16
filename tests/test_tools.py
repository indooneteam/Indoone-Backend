from __future__ import annotations

import asyncio
import json
import threading
import time

from app.ai.tools import MAX_CALCULATOR_ABS_VALUE, MAX_CALCULATOR_EXPONENT, TOOLS, _run_async, run_tool


def test_text_stats_tool_is_deterministic() -> None:
    result = run_tool("text_stats", "hello world\nagain")
    assert result.safe is True
    assert result.truncated is False
    assert json.loads(result.output) == {"characters": 17, "lines": 2, "words": 3}


def test_json_summary_tool_is_bounded() -> None:
    result = run_tool("json_summary", '{"name":"Indoone","version":2}')
    assert result.safe is True
    assert json.loads(result.output) == {"key_names": ["name", "version"], "keys": 2, "type": "object"}


def test_unknown_tool_is_unsafe() -> None:
    result = run_tool("not_a_tool", "payload")
    assert result.safe is False


def test_calculator_rejects_huge_exponent() -> None:
    result = run_tool("calculator", f"2**{MAX_CALCULATOR_EXPONENT + 1}")
    assert result.safe is False


def test_calculator_rejects_huge_literal() -> None:
    result = run_tool("calculator", str(MAX_CALCULATOR_ABS_VALUE + 1))
    assert result.safe is False


def test_tool_payload_limit_is_enforced() -> None:
    result = run_tool("text_stats", "x" * 100_001)
    assert result.safe is False
    assert "payload" in result.output.lower()


def test_tool_name_is_normalized() -> None:
    result = run_tool(" TEXT_STATS ", "hello")
    assert result.safe is True
    assert result.name == "text_stats"


def test_non_string_payload_is_rejected() -> None:
    result = run_tool("text_stats", None)  # type: ignore[arg-type]
    assert result.safe is False
    assert "payload" in result.output.lower()


def test_sync_tool_timeout_is_bounded(monkeypatch) -> None:
    original = TOOLS["text_stats"]

    def slow_tool(_: str) -> str:
        time.sleep(0.2)
        return "done"

    monkeypatch.setitem(TOOLS, "text_stats", slow_tool)
    monkeypatch.setattr("app.ai.tools.get_tool_spec", lambda _: type("Spec", (), {"timeout_seconds": 0.05, "max_output_chars": 100})())
    started = time.monotonic()
    result = run_tool("text_stats", "hello")
    elapsed = time.monotonic() - started
    monkeypatch.setitem(TOOLS, "text_stats", original)
    assert result.safe is False
    assert result.retryable is True
    assert "timed out" in result.output.lower()
    assert elapsed < 0.15


def test_total_tool_run_budget_is_enforced(monkeypatch) -> None:
    original = TOOLS["text_stats"]

    def slow_tool(_: str) -> str:
        time.sleep(0.2)
        return "done"

    monkeypatch.setitem(TOOLS, "text_stats", slow_tool)
    monkeypatch.setattr("app.ai.tools.MAX_TOOL_RUN_SECONDS", 0.05)
    monkeypatch.setattr("app.ai.tools.get_tool_spec", lambda _: type("Spec", (), {"timeout_seconds": 1.0, "max_output_chars": 100})())
    started = time.monotonic()
    result = run_tool("text_stats", "hello")
    elapsed = time.monotonic() - started
    monkeypatch.setitem(TOOLS, "text_stats", original)
    assert result.safe is False
    assert result.retryable is True
    assert "timed out" in result.output.lower()
    assert elapsed < 0.15


def test_tool_output_truncation_is_explicit(monkeypatch) -> None:
    original = TOOLS["text_stats"]
    monkeypatch.setitem(TOOLS, "text_stats", lambda _: "abcdefghij")
    monkeypatch.setattr("app.ai.tools.get_tool_spec", lambda _: type("Spec", (), {"timeout_seconds": 1.0, "max_output_chars": 4})())
    result = run_tool("text_stats", "hello")
    monkeypatch.setitem(TOOLS, "text_stats", original)
    assert result.safe is True
    assert result.truncated is True
    assert result.output == "abcd\n[output truncated by policy]"


def test_non_text_tool_output_is_unsafe(monkeypatch) -> None:
    original = TOOLS["text_stats"]
    monkeypatch.setitem(TOOLS, "text_stats", lambda _: 123)  # type: ignore[assignment]
    result = run_tool("text_stats", "hello")
    monkeypatch.setitem(TOOLS, "text_stats", original)
    assert result.safe is False
    assert result.retryable is False
    assert "output must be text" in result.output.lower()


def test_sync_tool_worker_slots_are_bounded(monkeypatch) -> None:
    original_tool = TOOLS["text_stats"]

    def blocked_tool(_: str) -> str:
        time.sleep(0.2)
        return "done"

    monkeypatch.setitem(TOOLS, "text_stats", blocked_tool)
    monkeypatch.setattr("app.ai.tools.get_tool_spec", lambda _: type("Spec", (), {"timeout_seconds": 0.2, "max_output_chars": 100})())
    results: list[object] = []
    lock = threading.Lock()

    def call() -> None:
        outcome = run_tool("text_stats", "hello")
        with lock:
            results.append(outcome)

    workers = [threading.Thread(target=call) for _ in range(9)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    monkeypatch.setitem(TOOLS, "text_stats", original_tool)

    unsafe = [item for item in results if getattr(item, "safe", True) is False]
    assert len(results) == 9
    assert any("too many" in getattr(item, "output", "").lower() for item in unsafe)


def test_async_timeout_cancels_coroutine_without_background_thread() -> None:
    cancelled = False

    async def slow() -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(1)
            return "done"
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def scenario() -> None:
        try:
            await _run_async(slow(), 0.01)
        except asyncio.TimeoutError:
            pass

    asyncio.run(scenario())
    assert cancelled is True


def test_async_run_stays_on_current_loop() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        async def current_loop() -> bool:
            return asyncio.get_running_loop() is loop

        assert await _run_async(current_loop(), 0.2) is True

    asyncio.run(scenario())
