from __future__ import annotations

import json

import pytest

from app.ai.code_sandbox import build_sandbox_request, sandbox_execution_tool
from app.ai.agent import build_agent_steps, execute_agent


def test_sandbox_request_is_disabled_by_default() -> None:
    result = build_sandbox_request("python", "print(1)")
    assert result["execution"] == "disabled"
    assert result["timeout_seconds"] == 5
    assert result["max_output_bytes"] == 16_384
    assert result["max_memory_mb"] == 128


def test_sandbox_rejects_invalid_language_and_empty_code() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        build_sandbox_request("brainfuck", "++")
    with pytest.raises(ValueError, match="must not be empty"):
        build_sandbox_request("python", "   ")


def test_sandbox_tool_returns_json_contract() -> None:
    payload = json.dumps({"language": "python", "code": "print('hello')"})
    result = json.loads(sandbox_execution_tool(payload))
    assert result["execution"] == "disabled"
    assert result["language"] == "python"


def test_agent_requires_approval_for_sandbox_execution() -> None:
    steps = build_agent_steps("run code: print('hello')")
    assert len(steps) == 1
    assert steps[0].tool == "sandbox_execution"
    assert steps[0].requires_approval is True


def test_agent_blocks_sandbox_without_approval() -> None:
    execution = execute_agent("run code: {\"language\": \"python\", \"code\": \"print(1)\"}")
    assert execution.results == ()
    assert len(execution.blocked_steps) == 1
    assert execution.blocked_steps[0].tool == "sandbox_execution"
