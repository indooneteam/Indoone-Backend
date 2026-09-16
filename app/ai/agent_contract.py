from __future__ import annotations

from typing import Any

from app.ai.agent import AgentExecution, AgentStep
from app.ai.tools import ToolResult

CONTRACT_VERSION = "agent.v2"


def serialize_step(step: AgentStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "tool": step.tool,
        "payload": step.payload,
        "requires_approval": step.requires_approval,
    }


def serialize_result(result: ToolResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "output": result.output,
        "safe": result.safe,
        "retryable": result.retryable,
        "truncated": result.truncated,
    }


def serialize_execution(execution: AgentExecution) -> dict[str, Any]:
    first = execution.steps[0] if execution.steps else (execution.blocked_steps[0] if execution.blocked_steps else None)
    first_result = execution.results[0] if execution.results else None
    return {
        "contract_version": CONTRACT_VERSION,
        "intent": "tool" if first is not None else "general",
        "tool": None if first is None else first.tool,
        "tool_payload": None if first is None else first.payload,
        "result": None if first_result is None else serialize_result(first_result),
        "steps": [serialize_step(step) for step in execution.steps],
        "results": [serialize_result(result) for result in execution.results],
        "memories": list(execution.memories),
        "blocked_steps": [serialize_step(step) for step in execution.blocked_steps],
        "retry_counts": list(execution.retry_counts),
        "run_id": execution.run_id,
    }
