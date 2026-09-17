from __future__ import annotations

import time

from app.ai.tools import ToolResult


def remaining_budget(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def should_retry(result: ToolResult, retry_count: int, max_retries: int) -> bool:
    return retry_count < max_retries and not result.safe and result.retryable


def deadline_failure(tool: str) -> ToolResult:
    return ToolResult(tool, "Tool execution deadline exceeded", safe=False, retryable=False)
