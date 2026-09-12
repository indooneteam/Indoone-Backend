from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: str = "read"
    requires_approval: bool = False
    timeout_seconds: float = 20.0
    max_output_chars: int = 20_000
    categories: tuple[str, ...] = ()


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("calculator", "Evaluate safe numeric arithmetic.", categories=("reasoning",)),
    ToolSpec("text_stats", "Count text characters, words, and lines.", categories=("text",)),
    ToolSpec("json_summary", "Summarize the top-level JSON structure.", categories=("data",)),
    ToolSpec("code_analysis", "Perform safe static code analysis.", categories=("coding",)),
    ToolSpec("code_fix_suggestions", "Produce static repair guidance for code.", categories=("coding",)),
    ToolSpec("code_transform", "Apply deterministic code transformations.", categories=("coding",)),
    ToolSpec("sandbox_execution", "Execute guarded code in the sandbox.", risk="execution", requires_approval=True, categories=("coding", "execution")),
    ToolSpec("contacts_search", "Search a client-authorized contact snapshot.", categories=("contacts",)),
    ToolSpec("contact_resolve", "Resolve a client-authorized contact.", categories=("contacts",)),
    ToolSpec("phone_call", "Prepare a phone call action.", risk="external", requires_approval=True, categories=("phone", "external")),
    ToolSpec("phone_call_contact", "Prepare a phone call action for a contact.", risk="external", requires_approval=True, categories=("phone", "external")),
    ToolSpec("gmail_search", "Search the authenticated user's Gmail mailbox.", categories=("gmail", "external")),
    ToolSpec("gmail_read", "Read one message from the authenticated user's Gmail mailbox.", categories=("gmail", "external")),
)


def get_tool_spec(name: str) -> ToolSpec | None:
    normalized = name.strip().lower()
    return next((item for item in TOOL_SPECS if item.name == normalized), None)


def tool_names() -> frozenset[str]:
    return frozenset(item.name for item in TOOL_SPECS)


def auto_approved_tool_names() -> frozenset[str]:
    return frozenset(item.name for item in TOOL_SPECS if not item.requires_approval)


def requires_approval(name: str) -> bool:
    spec = get_tool_spec(name)
    return True if spec is None else spec.requires_approval


def describe_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "description": item.description,
            "risk": item.risk,
            "requires_approval": item.requires_approval,
            "timeout_seconds": item.timeout_seconds,
            "max_output_chars": item.max_output_chars,
            "categories": list(item.categories),
        }
        for item in TOOL_SPECS
    ]
