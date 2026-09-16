import time

from app.ai.agent_policy import deadline_failure, remaining_budget, should_retry
from app.ai.tools import ToolResult


def test_remaining_budget_is_non_negative() -> None:
    deadline = time.monotonic() - 1
    assert remaining_budget(deadline) == 0.0


def test_should_retry_requires_retryable_unsafe_result_and_budget() -> None:
    assert should_retry(ToolResult("x", "temporary", safe=False, retryable=True), 0, 1) is True
    assert should_retry(ToolResult("x", "safe", safe=True, retryable=True), 0, 1) is False
    assert should_retry(ToolResult("x", "permanent", safe=False, retryable=False), 0, 1) is False
    assert should_retry(ToolResult("x", "temporary", safe=False, retryable=True), 1, 1) is False


def test_deadline_failure_is_unsafe_and_non_retryable() -> None:
    result = deadline_failure("calculator")
    assert result.safe is False
    assert result.retryable is False
    assert result.name == "calculator"
