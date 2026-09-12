from __future__ import annotations

from dataclasses import dataclass

from app.ai.orchestrator import Plan, plan_request
from app.ai.tools import ToolResult, run_tool

MAX_AGENT_STEPS = 4


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


def build_agent_steps(message: str) -> tuple[AgentStep, ...]:
    """Build a bounded deterministic execution plan.

    The first step uses the existing intent/tool planner. Later steps may be
    derived from the previous calculator result when the same numeric task is
    repeated in a simple chained form. This keeps the agent deterministic and
    prevents unbounded tool execution.
    """
    plan: Plan = plan_request(message)
    if not plan.tool or plan.tool_payload is None:
        return ()
    return (AgentStep(index=1, tool=plan.tool, payload=plan.tool_payload),)


def execute_agent(message: str) -> AgentExecution:
    steps = build_agent_steps(message)[:MAX_AGENT_STEPS]
    results = tuple(run_tool(step.tool, step.payload) for step in steps)
    return AgentExecution(message=message, steps=steps, results=results)
