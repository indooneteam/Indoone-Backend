from __future__ import annotations

from app.ai.tools import ToolResult


def synthesize_tool_answer(results: tuple[ToolResult, ...] | list[ToolResult]) -> str:
    """Create a bounded user-facing answer from tool execution results.

    This is intentionally deterministic: the tool output is preserved while
    unsafe or missing results are converted into a safe failure message.
    """
    if not results:
        return ""
    result = results[-1]
    if not result.safe:
        return "I’m sorry, the requested action could not be completed safely."
    return result.output.strip()
