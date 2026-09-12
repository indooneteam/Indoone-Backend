from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.ai.orchestrator import Plan, plan_request
from app.ai.tools import ToolResult, run_tool
from app.capabilities.store import create_agent_run, search_memories, update_agent_run

MAX_AGENT_STEPS = 4
MAX_AGENT_MEMORIES = 5
MAX_AGENT_RETRIES = 1
MAX_AGENT_MESSAGE_LENGTH = 20_000
MAX_AGENT_PAYLOAD_LENGTH = 8_000
ALLOWED_AGENT_TOOLS = frozenset({"calculator", "text_stats", "json_summary", "code_analysis", "code_fix_suggestions", "code_transform", "sandbox_execution"})
AUTO_APPROVED_AGENT_TOOLS = frozenset({"calculator", "text_stats", "json_summary", "code_analysis", "code_fix_suggestions", "code_transform"})


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


def _step(tool: str, payload: str, index: int) -> AgentStep:
    if len(payload) > MAX_AGENT_PAYLOAD_LENGTH:
        payload = payload[:MAX_AGENT_PAYLOAD_LENGTH]
    return AgentStep(index=index, tool=tool, payload=payload, requires_approval=tool not in AUTO_APPROVED_AGENT_TOOLS)


def _extract_explicit_requests(message: str) -> list[tuple[int, str, str]]:
    patterns = (
        ("sandbox_execution", r"(?:run|execute|sandbox)\s+code\s*:\s*(.+)$"),
        ("code_transform", r"(?:transform|normalize|format)\s+code\s*:\s*(.+)$"),
        ("code_fix_suggestions", r"(?:fix suggestions|suggest fixes|debug code)\s*:\s*(.+)$"),
        ("code_analysis", r"(?:code analysis|analyze code)\s*:\s*(.+)$"),
        ("json_summary", r"(?:summarize|summary of)\s+json\s*:\s*(\{.*?\}|\[.*?\])"),
        ("text_stats", r"(?:text stats|analyze text|count words)\s*:\s*(.+?)(?=\s*(?:;|$))"),
    )
    matches: list[tuple[int, str, str]] = []
    for tool, pattern in patterns:
        for match in re.finditer(pattern, message, re.IGNORECASE | re.DOTALL):
            matches.append((match.start(), tool, match.group(1).strip()))
    matches.sort(key=lambda item: item[0])
    return matches[:MAX_AGENT_STEPS]


def _extract_calculations(message: str) -> tuple[str, ...]:
    operand = r"(?:\$last|\$result\d+|\d+(?:\.\d+)?)"
    expression = rf"(?<!\w){operand}(?:\s*[+\-*/%]\s*{operand})+(?!\w)"
    matches = re.findall(expression, message)
    return tuple(match.replace(" ", "") for match in matches[:MAX_AGENT_STEPS])


def build_agent_steps(message: str) -> tuple[AgentStep, ...]:
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_AGENT_MESSAGE_LENGTH:
        return ()
    explicit = _extract_explicit_requests(normalized)
    if explicit:
        return tuple(_step(tool, payload, index) for index, (_, tool, payload) in enumerate(explicit, start=1))
    expressions = _extract_calculations(normalized)
    if expressions:
        return tuple(_step("calculator", expression, index) for index, expression in enumerate(expressions, start=1))
    plan: Plan = plan_request(normalized)
    if not plan.tool or plan.tool_payload is None or plan.tool not in ALLOWED_AGENT_TOOLS:
        return ()
    return (_step(plan.tool, plan.tool_payload, 1),)


def _load_memory_context(user_id: str, message: str) -> tuple[dict[str, Any], ...]:
    if not user_id.strip():
        return ()
    return tuple(search_memories(user_id, message[:MAX_AGENT_PAYLOAD_LENGTH], limit=MAX_AGENT_MEMORIES))


def _chain_payload(payload: str, results: tuple[ToolResult, ...]) -> str:
    if not results:
        return payload[:MAX_AGENT_PAYLOAD_LENGTH]
    chained = payload.replace("$last", results[-1].output)
    for index, result in enumerate(results, start=1):
        chained = chained.replace(f"$result{index}", result.output)
    return chained[:MAX_AGENT_PAYLOAD_LENGTH]


def _run_with_retry(tool: str, payload: str) -> tuple[ToolResult, int]:
    result = run_tool(tool, payload)
    if result.safe or MAX_AGENT_RETRIES == 0:
        return result, 0
    return run_tool(tool, payload), 1


def _serialize_step(step: AgentStep) -> dict[str, Any]:
    return {"index": step.index, "tool": step.tool, "payload": step.payload, "requires_approval": step.requires_approval}


def _serialize_result(result: ToolResult) -> dict[str, Any]:
    return {"name": result.name, "output": result.output, "safe": result.safe}


def execute_agent(message: str, user_id: str = "", approved_tools: set[str] | frozenset[str] | None = None) -> AgentExecution:
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_AGENT_MESSAGE_LENGTH:
        return AgentExecution(message=message, steps=(), results=())
    approved = frozenset(approved_tools or ())
    memory_context = _load_memory_context(user_id, normalized)
    run_id: str | None = None
    if user_id.strip():
        run_id = str(create_agent_run(user_id, normalized)["id"])
    planned_steps = build_agent_steps(normalized)[:MAX_AGENT_STEPS]
    executable: list[AgentStep] = []
    blocked: list[AgentStep] = []
    results: list[ToolResult] = []
    retry_counts: list[int] = []
    for step in planned_steps:
        if step.tool not in ALLOWED_AGENT_TOOLS or (step.requires_approval and step.tool not in approved):
            blocked.append(step)
            continue
        executable.append(step)
        result, retry_count = _run_with_retry(step.tool, _chain_payload(step.payload, tuple(results)))
        results.append(result)
        retry_counts.append(retry_count)
    unsafe_result = any(not result.safe for result in results)
    status = "failed" if unsafe_result else ("blocked" if blocked and not results else "completed")
    execution = AgentExecution(message=normalized, steps=tuple(executable), results=tuple(results), memories=memory_context, blocked_steps=tuple(blocked), retry_counts=tuple(retry_counts), run_id=run_id)
    if user_id.strip() and run_id is not None:
        update_agent_run(user_id=user_id, run_id=run_id, status=status, steps=[_serialize_step(step) for step in execution.steps], results=[_serialize_result(result) for result in execution.results], blocked_steps=[_serialize_step(step) for step in execution.blocked_steps], retry_counts=list(execution.retry_counts))
    return execution
