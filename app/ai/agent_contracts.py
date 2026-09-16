from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.ai.tool_registry import get_tool_spec
from app.ai.tools import ToolResult


@dataclass(frozen=True)
class ToolInvocation:
    tool: str
    payload: str
    index: int

    def __post_init__(self) -> None:
        normalized = self.tool.strip().lower()
        if not normalized:
            raise ValueError("tool is required")
        if not self.payload:
            raise ValueError("payload is required")
        if self.index < 1:
            raise ValueError("index must be >= 1")
        if get_tool_spec(normalized) is None:
            raise ValueError(f"unknown tool: {normalized}")
        object.__setattr__(self, "tool", normalized)


@dataclass(frozen=True)
class ToolExecutionContract:
    result: ToolResult
    retry_count: int = 0
    elapsed_ms: int | None = None

    def __post_init__(self) -> None:
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if self.elapsed_ms is not None and self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self.result)
        data["retry_count"] = self.retry_count
        data["elapsed_ms"] = self.elapsed_ms
        return data


def serialize_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "output": result.output,
        "safe": result.safe,
        "retryable": result.retryable,
        "truncated": result.truncated,
    }
