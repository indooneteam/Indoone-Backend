from __future__ import annotations

from dataclasses import dataclass
import re

from app.ai.intent import Intent, classify_intent
from app.ai.tools import ToolResult, run_tool


@dataclass(frozen=True)
class Plan:
    intent: Intent
    tool: str | None = None
    tool_payload: str | None = None


def _extract_arithmetic(message: str) -> str | None:
    matches = re.findall(r"(?<!\w)(?:\d+(?:\.\d+)?\s*[+\-*/%]\s*)+\d+(?:\.\d+)?(?!\w)", message)
    return matches[0].replace(" ", "") if matches else None


def plan_request(message: str) -> Plan:
    intent = classify_intent(message)
    if intent.needs_calculation:
        expression = _extract_arithmetic(message)
        if expression:
            return Plan(intent=intent, tool="calculator", tool_payload=expression)
    return Plan(intent=intent)


def execute_plan(plan: Plan) -> ToolResult | None:
    if not plan.tool or plan.tool_payload is None:
        return None
    return run_tool(plan.tool, plan.tool_payload)
