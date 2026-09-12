from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int = 60_000
    max_history_messages: int = 40
    max_memory_items: int = 10
    max_tool_output_chars: int = 20_000


def trim_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def trim_history(history: list[tuple[str, str]], budget: ContextBudget) -> list[tuple[str, str]]:
    selected = history[-budget.max_history_messages :]
    result: list[tuple[str, str]] = []
    used = 0
    for role, content in reversed(selected):
        remaining = budget.max_chars - used
        if remaining <= 0:
            break
        clipped = trim_text(content, min(remaining, 12_000))
        result.append((role, clipped))
        used += len(clipped)
    return list(reversed(result))
