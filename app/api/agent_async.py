from __future__ import annotations

from fastapi import APIRouter

from app.ai.agent import MAX_AGENT_STEPS
from app.ai.agent_async import execute_agent_async
from app.api.capabilities import AgentRequest

router = APIRouter(tags=["capabilities"])


@router.post("/agent")
async def agent(request: AgentRequest) -> dict[str, object]:
    execution = await execute_agent_async(
        request.message,
        user_id=request.user_id,
        approved_tools=frozenset(request.approved_tools),
        contacts=request.contacts,
    )
    if not execution.steps and not execution.blocked_steps:
        return {
            "intent": "general",
            "tool": None,
            "tool_payload": None,
            "result": None,
            "steps": [],
            "results": [],
            "memories": list(execution.memories),
            "blocked_steps": [],
            "retry_counts": list(execution.retry_counts),
            "run_id": execution.run_id,
            "max_steps": MAX_AGENT_STEPS,
        }

    first = execution.steps[0] if execution.steps else execution.blocked_steps[0]
    first_result = execution.results[0] if execution.results else None

    def serialize_result(result):
        return {
            "name": result.name,
            "output": result.output,
            "safe": result.safe,
            "retryable": result.retryable,
            "truncated": result.truncated,
        }

    return {
        "intent": "tool",
        "tool": first.tool,
        "tool_payload": first.payload,
        "result": None if first_result is None else serialize_result(first_result),
        "steps": [
            {
                "index": step.index,
                "tool": step.tool,
                "payload": step.payload,
                "requires_approval": step.requires_approval,
            }
            for step in execution.steps
        ],
        "results": [serialize_result(result) for result in execution.results],
        "memories": list(execution.memories),
        "blocked_steps": [
            {
                "index": step.index,
                "tool": step.tool,
                "payload": step.payload,
                "requires_approval": step.requires_approval,
            }
            for step in execution.blocked_steps
        ],
        "retry_counts": list(execution.retry_counts),
        "run_id": execution.run_id,
        "max_steps": MAX_AGENT_STEPS,
    }
