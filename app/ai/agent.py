from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from app.ai.approval import decide_tool
from app.ai.orchestrator import Plan, plan_request
from app.ai.tool_registry import get_tool_spec, tool_names
from app.ai.tools import ToolResult, run_tool
from app.capabilities.store import create_agent_run, search_memories, update_agent_run

MAX_AGENT_STEPS = 4
MAX_AGENT_MEMORIES = 5
MAX_AGENT_RETRIES = 1
MAX_AGENT_MESSAGE_LENGTH = 20_000
MAX_AGENT_PAYLOAD_LENGTH = 8_000
ALLOWED_AGENT_TOOLS = tool_names()


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
    spec = get_tool_spec(tool)
    return AgentStep(
        index=index,
        tool=tool,
        payload=payload,
        requires_approval=True if spec is None else spec.requires_approval,
    )


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


def _gmail_request_matches(message: str) -> bool:
    text = message.strip()
    patterns = (
        r"(?:search|find|list)\s+(?:in\s+)?gmail(?:\s+for)?\s+.+$",
        r"gmail\s+(?:search|find|list)\s+.+$",
        r"(?:read|open|show)\s+(?:gmail\s+)?(?:email|message)\s+[A-Za-z0-9_-]+$",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _extract_gmail_request(message: str, user_id: str) -> tuple[str, dict[str, Any]] | None:
    if not user_id.strip():
        return None
    text = message.strip()
    patterns = (
        ("gmail_search", r"(?:search|find|list)\s+(?:in\s+)?gmail(?:\s+for)?\s+(.+)$"),
        ("gmail_search", r"gmail\s+(?:search|find|list)\s+(.+)$"),
        ("gmail_read", r"(?:read|open|show)\s+(?:gmail\s+)?(?:email|message)\s+([A-Za-z0-9_-]+)$"),
    )
    for tool, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = match.group(1).strip(" .?!")
            if query:
                if tool == "gmail_search":
                    return tool, {"user_id": user_id, "query": query}
                return tool, {"user_id": user_id, "message_id": query}
    return None


def _extract_contact_request(message: str, contacts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    text = message.strip()
    patterns = (
        ("phone_call_contact", r"(?:call|phone|dial|ಕರೆ|ಫೋನ್|ಕರೆಮಾಡು)\s+(?:to\s+|ge\s+|ಗೆ\s+|maadu\s+|ಮಾಡು\s+)?(.+)$"),
        ("contact_resolve", r"(?:find\s+contact|search\s+contact|lookup\s+contact|resolve\s+contact|contact|find|search|lookup|resolve|ಹುಡುಕು|ಕಾಂಟ್ಯಾಕ್ಟ್)\s+(.+)$"),
    )
    for tool, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = match.group(1).strip(" .?!")
            if query:
                return tool, {"contacts": contacts, "query": query}
    return None


def build_agent_steps(message: str, user_id: str = "", contacts: list[dict[str, Any]] | None = None) -> tuple[AgentStep, ...]:
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_AGENT_MESSAGE_LENGTH:
        return ()
    explicit = _extract_explicit_requests(normalized)
    if explicit:
        return tuple(_step(tool, payload, index) for index, (_, tool, payload) in enumerate(explicit, start=1))
    gmail_request = _extract_gmail_request(normalized, user_id)
    if gmail_request is not None:
        tool, payload = gmail_request
        return (_step(tool, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), 1),)
    if _gmail_request_matches(normalized):
        return ()
    contact_request = _extract_contact_request(normalized, contacts or [])
    if contact_request is not None:
        tool, payload = contact_request
        return (_step(tool, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), 1),)
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


def execute_agent(
    message: str,
    user_id: str = "",
    approved_tools: set[str] | frozenset[str] | None = None,
    contacts: list[dict[str, Any]] | None = None,
    approval_tokens: Iterable[str] | None = None,
) -> AgentExecution:
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_AGENT_MESSAGE_LENGTH:
        return AgentExecution(message=message, steps=(), results=())
    # approved_tools is kept for compatibility only; authorization is cryptographically
    # bound to the authenticated user through server-issued approval tokens.
    approved = frozenset(approved_tools or ())
    approval_tokens_tuple = tuple(approval_tokens or ())
    memory_context = _load_memory_context(user_id, normalized)
    run_id: str | None = None
    if user_id.strip():
        run_id = str(create_agent_run(user_id, normalized)["id"])
    planned_steps = build_agent_steps(normalized, user_id=user_id, contacts=contacts)[:MAX_AGENT_STEPS]
    executable: list[AgentStep] = []
    blocked: list[AgentStep] = []
    results: list[ToolResult] = []
    retry_counts: list[int] = []
    for step in planned_steps:
        decision = decide_tool(step.tool, approved, approval_tokens_tuple, user_id=user_id)
        if not decision.allowed:
            blocked.append(step)
            continue
        executable.append(step)
        result, retry_count = _run_with_retry(step.tool, _chain_payload(step.payload, tuple(results)))
        spec = get_tool_spec(step.tool)
        if spec is not None and len(result.output) > spec.max_output_chars:
            result = ToolResult(step.tool, result.output[:spec.max_output_chars], safe=False)
        results.append(result)
        retry_counts.append(retry_count)
    unsafe_result = any(not result.safe for result in results)
    status = "failed" if unsafe_result else ("blocked" if blocked and not results else "completed")
    execution = AgentExecution(message=normalized, steps=tuple(executable), results=tuple(results), memories=memory_context, blocked_steps=tuple(blocked), retry_counts=tuple(retry_counts), run_id=run_id)
    if user_id.strip() and run_id is not None:
        update_agent_run(user_id=user_id, run_id=run_id, status=status, steps=[_serialize_step(step) for step in execution.steps], results=[_serialize_result(result) for result in execution.results], blocked_steps=[_serialize_step(step) for step in execution.blocked_steps], retry_counts=list(execution.retry_counts))
    return execution
