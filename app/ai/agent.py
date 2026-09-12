from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.ai.orchestrator import Plan, plan_request
from app.ai.tools import ToolResult, run_tool
from app.capabilities.store import (
    create_agent_run,
    search_memories,
    update_agent_run,
)

MAX_AGENT_STEPS = 4
MAX_AGENT_MEMORIES = 5
MAX_AGENT_RETRIES = 1
ALLOWED_AGENT_TOOLS = frozenset({"calculator"})
AUTO_APPROVED_AGENT_TOOLS = frozenset({"calculator"})


@dataclass(frozen=True)
class AgentStep:
    index: int
    tool: str
    payload: str
    requires_approval: bool = False


@dataclass(frozen=True)
class AgentExecution:
    message: str
    steps: tuple[AgentStep, ...]
    results: tuple[ToolResult, ...]
    memories: tuple[dict[str, Any], ...] = ()
    blocked_steps: tuple[AgentStep, ...] = ()
    retry_counts: tuple[int, ...] = ()
    run_id: str | None = None


def _extract_calculations(message: str) -> tuple[str, ...]:
    operand = r"(?:\$last|\$result\d+|\d+(?:\.\d+)?)"
    expression = rf"(?<!\w){operand}(?:\s*[+\-*/%]\s*{operand})+(?!\w)"
    matches = re.findall(expression, message)
    return tuple(match.replace(" ", "") for match in matches[:MAX_AGENT_STEPS])


def build_agent_steps(message: str) -> tuple[AgentStep, ...]:
    """Build a bounded deterministic multi-step execution plan."""
    expressions = _extract_calculations(message)
    if expressions:
        return tuple(
            AgentStep(
                index=index,
                tool="calculator",
                payload=expression,
                requires_approval="calculator" not in AUTO_APPROVED_AGENT_TOOLS,
            )
            for index, expression in enumerate(expressions, start=1)
        )

    plan: Plan = plan_request(message)
    if not plan.tool or plan.tool_payload is None:
        return ()
    if plan.tool not in ALLOWED_AGENT_TOOLS:
        return ()
    return (
        AgentStep(
            index=1,
            tool=plan.tool,
            payload=plan.tool_payload,
            requires_approval=plan.tool not in AUTO_APPROVED_AGENT_TOOLS,
        ),
    )


def _load_memory_context(user_id: str, message: str) -> tuple[dict[str, Any], ...]:
    if not user_id.strip():
        return ()
    return tuple(search_memories(user_id, message, limit=MAX_AGENT_MEMORIES))


def _chain_payload(payload: str, results: tuple[ToolResult, ...]) -> str:
    """Resolve bounded result placeholders before a dependent tool call."""
    if not results:
        return payload
    chained = payload.replace("$last", results[-1].output)
    for index, result in enumerate(results, start=1):
        chained = chained.replace(f"$result{index}", result.output)
    return chained


def _run_with_retry(tool: str, payload: str) -> tuple[ToolResult, int]:
    """Retry a failed tool once with the exact same bounded payload."""
    result = run_tool(tool, payload)
    if result.safe or MAX_AGENT_RETRIES == 0:
        return result, 0
    retry_result = run_tool(tool, payload)
    return retry_result, 1


def _serialize_step(step: AgentStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "tool": step.tool,
        "payload": step.payload,
        "requires_approval": step.requires_approval,
    }


def _serialize_result(result: ToolResult) -> dict[str, Any]:
    return {"name": result.name, "output": result.output, "safe": result.safe}


def execute_agent(
    message: str,
    user_id: str = "",
    approved_tools: set[str] | frozenset[str] | None = None,
) -> AgentExecution:
    """Execute only allowed/approved tools with bounded retries and persisted state."""
    approved = frozenset(approved_tools or ())
    memory_context = _load_memory_context(user_id, message)
    run_id: str | None = None
    if user_id.strip():
        run = create_agent_run(user_id, message)
        run_id = str(run["id"])

    planned_steps = build_agent_steps(message)[:MAX_AGENT_STEPS]
    executable: list[AgentStep] = []
    blocked: list[AgentStep] = []
    results: list[ToolResult] = []
    retry_counts: list[int] = []

    for step in planned_steps:
        if step.tool not in ALLOWED_AGENT_TOOLS:
            blocked.append(step)
            continue
        if step.requires_approval and step.tool not in approved:
            blocked.append(step)
            continue
        executable.append(step)
        chained_payload = _chain_payload(step.payload, tuple(results))
        result, retry_count = _run_with_retry(step.tool, chained_payload)
        results.append(result)
        retry_counts.append(retry_count)

    status = "blocked" if blocked and not results else "completed"
    execution = AgentExecution(
        message=message,
        steps=tuple(executable),
        results=tuple(results),
        memories=memory_context,
        blocked_steps=tuple(blocked),
        retry_counts=tuple(retry_counts),
        run_id=run_id,
    )

    if user_id.strip() and run_id is not None:
        update_agent_run(
            user_id=user_id,
            run_id=run_id,
            status=status,
            steps=[_serialize_step(step) for step in execution.steps],
            results=[_serialize_result(result) for result in execution.results],
            blocked_steps=[_serialize_step(step) for step in execution.blocked_steps],
            retry_counts=list(execution.retry_counts),
        )
    return execution
