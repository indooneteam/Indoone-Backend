from __future__ import annotations

import json

from app.ai.tools import ToolResult


def synthesize_tool_answer(results: tuple[ToolResult, ...] | list[ToolResult]) -> str:
    """Build a deterministic user-facing answer from successful tool results."""
    items = list(results)
    if not items:
        return "I’m sorry, I don’t have enough reliable information to give you a confident answer right now."

    safe_items = [item for item in items if item.safe]
    if not safe_items:
        return "I’m sorry, the requested tool action could not be completed safely."

    if len(safe_items) == 1:
        output = safe_items[0].output.strip()
        if not output:
            return "The tool completed without a user-facing result."
        return output

    payload = [
        {
            "tool": item.name,
            "output": item.output,
        }
        for item in safe_items
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
