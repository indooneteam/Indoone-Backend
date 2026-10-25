from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.coding import build_code_tool_payload
from app.ai.intent import Intent, classify_intent
from app.ai.tool_registry import get_tool_spec, tool_names
from app.ai.tools import ToolResult, run_tool


@dataclass(frozen=True)
class Plan:
    intent: Intent
    tool: str | None = None
    tool_payload: str | None = None
    reason: str = ""

    @property
    def executable(self) -> bool:
        return bool(self.tool and self.tool_payload is not None and get_tool_spec(self.tool) is not None)


def _extract_arithmetic(message: str) -> str | None:
    matches = re.findall(r"(?<!\w)(?:\d+(?:\.\d+)?\s*[+\-*/%]\s*)+\d+(?:\.\d+)?(?!\w)", message)
    return matches[0].replace(" ", "") if matches else None


def _extract_payload(message: str, prefixes: tuple[str, ...]) -> str | None:
    text = message.strip()
    lower = text.casefold()
    for prefix in prefixes:
        marker = prefix.casefold()
        if lower.startswith(marker):
            payload = text[len(prefix):].lstrip(" :")
            if payload:
                return payload
    return None


def _rule_plan(message: str, intent: Intent) -> Plan:
    text = message.strip()
    lower = text.casefold()

    if intent.needs_calculation:
        expression = _extract_arithmetic(text)
        if expression:
            return Plan(intent=intent, tool="calculator", tool_payload=expression, reason="arithmetic")

    # The last four are coding tools: the captured text is bare code, which the
    # executor does not accept, so it is wrapped in the {language, code} payload.
    candidates: tuple[tuple[str, tuple[str, ...], bool], ...] = (
        ("text_stats", ("text stats:", "analyze text:", "count words:", "count text:"), False),
        ("json_summary", ("summarize json:", "json summary:", "summary of json:"), False),
        ("code_analysis", ("code analysis:", "analyze code:", "review code:"), True),
        ("code_fix_suggestions", ("fix suggestions:", "suggest fixes:", "debug code:"), True),
        ("code_transform", ("transform code:", "normalize code:", "format code:"), True),
        ("sandbox_execution", ("run code:", "execute code:", "sandbox code:"), True),
    )
    for tool, prefixes, is_code in candidates:
        payload = _extract_payload(text, prefixes)
        if payload and tool in tool_names():
            if is_code:
                payload = build_code_tool_payload(payload)
            return Plan(intent=Intent("tool", needs_file_context=intent.needs_file_context), tool=tool, tool_payload=payload, reason="registry-rule")

    if lower.startswith(("search gmail", "find in gmail", "list gmail")):
        payload = _extract_payload(text, ("search gmail", "find in gmail", "list gmail"))
        if payload:
            return Plan(intent=Intent("tool"), tool="gmail_search", tool_payload=payload, reason="gmail-search")

    if lower.startswith(("read email ", "read gmail email ", "open email ")):
        payload = _extract_payload(text, ("read email", "read gmail email", "open email"))
        if payload:
            return Plan(intent=Intent("tool"), tool="gmail_read", tool_payload=payload, reason="gmail-read")

    return Plan(intent=intent, reason="no-safe-tool-match")


def plan_request(message: str) -> Plan:
    normalized = message.strip()
    if not normalized:
        return Plan(intent=classify_intent(message), reason="empty")
    intent = classify_intent(normalized)
    return _rule_plan(normalized, intent)


def execute_plan(plan: Plan) -> ToolResult | None:
    if not plan.executable:
        return None
    return run_tool(plan.tool or "", plan.tool_payload or "")
