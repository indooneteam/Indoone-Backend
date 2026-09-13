from __future__ import annotations

import ast
import asyncio
import json
import operator
import threading
from dataclasses import dataclass
from typing import Callable

from app.ai.coding import code_analysis_tool, code_fix_suggestions_tool, code_transform_tool
from app.ai.code_sandbox import sandbox_execution_tool
from app.ai.tool_registry import get_tool_spec
from app.capabilities.contacts import parse_contacts, resolve_contact, search_contacts
from app.capabilities.gmail import get_gmail_message, list_gmail_messages
from app.capabilities.phone import build_call_action

MAX_CALCULATOR_ABS_VALUE = 10**100
MAX_CALCULATOR_EXPONENT = 1000
MAX_TOOL_PAYLOAD = 100_000
MAX_TOOL_TIMEOUT_SECONDS = 60.0
MAX_SYNC_TOOL_WORKERS = 8


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str
    safe: bool = True
    retryable: bool = False


def _bounded_number(value: int | float) -> int | float:
    if abs(value) > MAX_CALCULATOR_ABS_VALUE:
        raise ValueError("calculator result is too large")
    return value


def _calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def walk(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return _bounded_number(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return _bounded_number(allowed[type(node.op)](walk(node.operand)))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                left = walk(node.left)
                exponent = walk(node.right)
                if abs(exponent) > MAX_CALCULATOR_EXPONENT or int(exponent) != exponent:
                    raise ValueError("calculator exponent is too large")
                return _bounded_number(operator.pow(left, exponent))
            if type(node.op) in allowed:
                return _bounded_number(allowed[type(node.op)](walk(node.left), walk(node.right)))
        raise ValueError("Only numeric arithmetic is allowed")

    return str(walk(tree.body))


def _text_stats(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    words = normalized.split()
    lines = normalized.splitlines() if normalized else []
    return json.dumps({"characters": len(text), "words": len(words), "lines": len(lines)}, sort_keys=True)


def _json_summary(payload: str) -> str:
    data = json.loads(payload)
    if isinstance(data, dict):
        return json.dumps({"type": "object", "keys": len(data), "key_names": sorted(str(key) for key in data)[:50]}, ensure_ascii=False, sort_keys=True)
    if isinstance(data, list):
        return json.dumps({"type": "array", "items": len(data)}, sort_keys=True)
    if data is None:
        return json.dumps({"type": "null"}, sort_keys=True)
    return json.dumps({"type": type(data).__name__}, sort_keys=True)


def _contacts_search(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("contacts_search payload must be an object")
    contacts = parse_contacts(data.get("contacts", []))
    matches = search_contacts(contacts, str(data.get("query", "")), int(data.get("limit", 20)))
    return json.dumps({"contacts": [item.as_dict() for item in matches], "count": len(matches)}, ensure_ascii=False, sort_keys=True)


def _contact_resolve(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("contact_resolve payload must be an object")
    contacts = parse_contacts(data.get("contacts", []))
    match = resolve_contact(contacts, str(data.get("query", "")))
    return json.dumps({"contact": match.as_dict() if match else None}, ensure_ascii=False, sort_keys=True)


def _phone_call(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("phone_call payload must be an object")
    action = build_call_action(data.get("phone_number"), data.get("contact_name", ""))
    return json.dumps(action.as_dict(), ensure_ascii=False, sort_keys=True)


def _phone_call_contact(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("phone_call_contact payload must be an object")
    contacts = parse_contacts(data.get("contacts", []))
    query = str(data.get("query", "")).strip()
    if not query:
        raise ValueError("contact query cannot be empty")
    match = resolve_contact(contacts, query)
    if match is None:
        return json.dumps({"contact": None, "requires_confirmation": False, "detail": "contact not found"}, ensure_ascii=False, sort_keys=True)
    action = build_call_action(match.phone, match.name)
    return json.dumps(action.as_dict(), ensure_ascii=False, sort_keys=True)


def _run_async(coro, timeout_seconds: float):
    bounded_timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coro, timeout=bounded_timeout))

    result: list[object] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(asyncio.wait_for(coro, timeout=bounded_timeout)))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True, name="indoone-async-tool")
    thread.start()
    thread.join(bounded_timeout + 0.5)
    if thread.is_alive():
        raise TimeoutError("asynchronous tool timed out")
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("asynchronous tool returned no result")
    return result[0]


_SYNC_TOOL_SEMAPHORE = threading.BoundedSemaphore(MAX_SYNC_TOOL_WORKERS)


def _run_sync_with_timeout(tool: Callable[[str], str], payload: str, timeout_seconds: float) -> str:
    bounded_timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    if not _SYNC_TOOL_SEMAPHORE.acquire(blocking=False):
        raise RuntimeError("too many timed-out or running sync tools")

    result: list[object] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(tool(payload))
        except BaseException as exc:
            errors.append(exc)
        finally:
            _SYNC_TOOL_SEMAPHORE.release()

    thread = threading.Thread(target=runner, daemon=True, name="indoone-sync-tool")
    thread.start()
    thread.join(bounded_timeout)
    if thread.is_alive():
        raise TimeoutError("tool execution timed out")
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("tool execution returned no result")
    output = result[0]
    if not isinstance(output, str):
        raise TypeError("tool output must be text")
    return output


def _gmail_search(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_search payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("gmail_search")
    result = _run_async(
        list_gmail_messages(
            user_id=user_id,
            query=str(data.get("query", "")),
            page_token=str(data.get("page_token", "")),
            max_results=int(data.get("max_results", 20)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _gmail_read(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_read payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    message_id = str(data.get("message_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not message_id:
        raise ValueError("message_id is required")
    spec = get_tool_spec("gmail_read")
    result = _run_async(get_gmail_message(user_id=user_id, message_id=message_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": _calculate,
    "text_stats": _text_stats,
    "json_summary": _json_summary,
    "code_analysis": code_analysis_tool,
    "code_fix_suggestions": code_fix_suggestions_tool,
    "code_transform": code_transform_tool,
    "sandbox_execution": sandbox_execution_tool,
    "contacts_search": _contacts_search,
    "contact_resolve": _contact_resolve,
    "phone_call": _phone_call,
    "phone_call_contact": _phone_call_contact,
    "gmail_search": _gmail_search,
    "gmail_read": _gmail_read,
}


def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError))


def run_tool(name: str, payload: str) -> ToolResult:
    normalized_name = str(name).strip().lower()
    if not isinstance(payload, str):
        return ToolResult(normalized_name, "Tool payload must be text", safe=False, retryable=False)
    if len(payload) > MAX_TOOL_PAYLOAD:
        return ToolResult(normalized_name, "Tool payload is too large", safe=False, retryable=False)
    spec = get_tool_spec(normalized_name)
    tool = TOOLS.get(normalized_name)
    if spec is None or tool is None:
        return ToolResult(normalized_name, "Unknown or unregistered tool", safe=False, retryable=False)
    try:
        output = _run_sync_with_timeout(tool, payload, spec.timeout_seconds)
        if len(output) > spec.max_output_chars:
            output = output[: spec.max_output_chars] + "\n[output truncated by policy]"
        return ToolResult(normalized_name, output, safe=True, retryable=False)
    except Exception as exc:
        return ToolResult(normalized_name, f"Tool error: {exc}", safe=False, retryable=_is_retryable_exception(exc))