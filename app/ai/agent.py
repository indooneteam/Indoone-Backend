from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.ai.orchestrator import Plan, plan_request
from app.ai.tools import ToolResult, run_tool
from app.capabilities.store import search_memories

MAX_AGENT_STEPS = 4
MAX_AGENT_MEMORIES = 5
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


def execute_agent(
    message: str,
    user_id: str = "",
    approved_tools: set[str] | frozenset[str] | None = None,
) -> AgentExecution:
    """Execute only allowed/approved tools and expose bounded memory context."""
    approved = frozenset(approved_tools or ())
    memory_context = _load_memory_context(user_id, message)
    planned_steps = build_agent_steps(message)[:MAX_AGENT_STEPS]
    executable: list[AgentStep] = []
    blocked: list[AgentStep] = []
    results: list[ToolResult] = []

    for step in planned_steps:
        if step.tool not in ALLOWED_AGENT_TOOLS:
            blocked.append(step)
            continue
        if step.requires_approval and step.tool not in approved:
            blocked.append(step)
            continue
        executable.append(step)
        chained_payload = _chain_payload(step.payload, tuple(results))
        results.append(run_tool(step.tool, chained_payload))

    return AgentExecution(
        message=message,
        steps=tuple(executable),
        results=tuple(results),
        memories=memory_context,
        blocked_steps=tuple(blocked),
    )
