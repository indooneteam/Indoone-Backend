from __future__ import annotations

import asyncio
import time
from typing import Iterable, Any

from app.ai.agent import (
    AgentExecution,
    MAX_AGENT_APPROVAL_TOKENS,
    MAX_AGENT_MESSAGE_LENGTH,
    MAX_AGENT_PAYLOAD_LENGTH,
    MAX_AGENT_RETRIES,
    MAX_AGENT_RUNTIME_SECONDS,
    _bounded_approval_tokens,
    _chain_payload,
    _load_memory_context,
    _serialize_result,
    _serialize_step,
    _step,
    _unresolved_chain_references,
    build_agent_steps,
)
from app.ai.approval import decide_tool
from app.ai.tools import ToolResult, run_tool_async
from app.capabilities.store import create_agent_run, update_agent_run


async def _run_with_retry_async(tool: str, payload: str, deadline: float) -> tuple[ToolResult, int]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return ToolResult(tool, "Tool execution deadline exceeded", safe=False, retryable=False), 0

    result = await run_tool_async(tool, payload, max_runtime_seconds=remaining)
    if result.safe or not result.retryable or MAX_AGENT_RETRIES == 0:
        return result, 0

    retry_count = 0
    while retry_count < MAX_AGENT_RETRIES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ToolResult(tool, "Tool execution deadline exceeded", safe=False, retryable=False), retry_count
        retry_count += 1
        result = await run_tool_async(tool, payload, max_runtime_seconds=remaining)
        if result.safe or not result.retryable:
            break
    return result, retry_count


async def execute_agent_async(
    message: str,
    user_id: str = "",
    approved_tools: set[str] | frozenset[str] | None = None,
    contacts: list[dict[str, Any]] | None = None,
    approval_tokens: Iterable[str] | None = None,
) -> AgentExecution:
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_AGENT_MESSAGE_LENGTH:
        return AgentExecution(message=message, steps=(), results=())

    approved = frozenset(approved_tools or ())
    approval_tokens_tuple = _bounded_approval_tokens(approval_tokens)
    memory_context = _load_memory_context(user_id, normalized)
    run_id: str | None = None
    if user_id.strip():
        run_id = str(create_agent_run(user_id, normalized)["id"])

    planned_steps = build_agent_steps(normalized, user_id=user_id, contacts=contacts)[:4]
    executable: list[Any] = []
    blocked: list[Any] = []
    results: list[ToolResult] = []
    retry_counts: list[int] = []
    deadline = time.monotonic() + MAX_AGENT_RUNTIME_SECONDS

    for position, step in enumerate(planned_steps):
        if time.monotonic() >= deadline:
            blocked.extend(planned_steps[position:])
            break

        decision = decide_tool(step.tool, approved, approval_tokens_tuple, user_id=user_id)
        if not decision.allowed:
            blocked.append(step)
            continue
        if results and not results[-1].safe:
            blocked.append(step)
            continue

        chain_results = tuple(results)
        chained_payload = _chain_payload(step.payload, chain_results)
        unresolved = _unresolved_chain_references(chained_payload, chain_results)
        if unresolved:
            blocked.append(step)
            result = ToolResult(step.tool, f"Unresolved chain reference(s): {', '.join(unresolved)}", safe=False, retryable=False)
            retry_count = 0
        elif chain_results and chain_results[-1].truncated:
            blocked.append(step)
            result = ToolResult(step.tool, "Cannot chain from truncated tool output", safe=False, retryable=False)
            retry_count = 0
        else:
            executable.append(step)
            result, retry_count = await _run_with_retry_async(step.tool, chained_payload, deadline)

        results.append(result)
        retry_counts.append(retry_count)
        if not result.safe:
            blocked.extend(planned_steps[position + 1 :])
            break

    unsafe_result = any(not result.safe for result in results)
    timed_out = bool(
        blocked
        and len(executable) + len(blocked) >= len(planned_steps)
        and time.monotonic() >= deadline
    )
    status = "failed" if unsafe_result else ("blocked" if blocked and not results else ("timed_out" if timed_out else "completed"))
    execution = AgentExecution(
        message=normalized,
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
