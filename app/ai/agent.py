from __future__ import annotations

from dataclasses import dataclass
import re

from app.ai.orchestrator import Plan, plan_request
from app.ai.tools import ToolResult, run_tool

MAX_AGENT_STEPS = 4
ALLOWED_AGENT_TOOLS = frozenset({"calculator"})


@dataclass(frozen=True)
class AgentStep:
    index: int
    tool: str
    payload: str


@dataclass(frozen=True)
class AgentExecution:
    message: str
    steps: tuple[AgentStep, ...]
    results: tuple[ToolResult, ...]


def _extract_calculations(message: str) -> tuple[str, ...]:
    matches = re.findall(
        r"(?<!\w)(?:\d+(?:\.\d+)?\s*[+\-*/%]\s*)+\d+(?:\.\d+)?(?!\w)",
        message,
    )
    return tuple(match.replace(" ", "") for match in matches[:MAX_AGENT_STEPS])


def build_agent_steps(message: str) -> tuple[AgentStep, ...]:
    """Build a bounded deterministic multi-step execution plan."""
    expressions = _extract_calculations(message)
    if expressions:
        return tuple(
            AgentStep(index=index, tool="calculator", payload=expression)
            for index, expression in enumerate(expressions, start=1)
        )

    plan: Plan = plan_request(message)
    if not plan.tool or plan.tool_payload is None:
        return ()
    if plan.tool not in ALLOWED_AGENT_TOOLS:
        return ()
    return (AgentStep(index=1, tool=plan.tool, payload=plan.tool_payload),)


def execute_agent(message: str) -> AgentExecution:
    steps = tuple(
        step for step in build_agent_steps(message)[:MAX_AGENT_STEPS]
        if step.tool in ALLOWED_AGENT_TOOLS
    )
    results = tuple(run_tool(step.tool, step.payload) for step in steps)
    return AgentExecution(message=message, steps=steps, results=results)
