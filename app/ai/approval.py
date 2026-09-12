from __future__ import annotations

from dataclasses import dataclass

from app.ai.tool_registry import get_tool_spec


@dataclass(frozen=True)
class ApprovalDecision:
    allowed: bool
    requires_approval: bool
    reason: str


def decide_tool(name: str, approved_tools: set[str] | frozenset[str] | None = None) -> ApprovalDecision:
    approved = frozenset(approved_tools or ())
    spec = get_tool_spec(name)
    if spec is None:
        return ApprovalDecision(False, True, "tool is not registered")
    if not spec.requires_approval:
        return ApprovalDecision(True, False, "tool is auto-approved by policy")
    if name in approved:
        return ApprovalDecision(True, True, "explicit approval supplied")
    return ApprovalDecision(False, True, "explicit approval is required")
