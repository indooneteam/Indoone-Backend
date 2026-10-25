"""Prove a GitHub write happened before anything says it did.

A write returning ok is the provider saying it accepted the call. It is not the same as
the change being there, and the difference is exactly what a user notices: "it said the
push went through and it had not". So every write reads the state back with an ordinary
read tool and reports what it saw.

The result is three fields, and all three are always present:

* ``verified``      true only when the read-back found the change;
* ``checked``       whether a read-back ran at all;
* ``verification``  one sentence a person can read, saying what was seen.

``verified`` is false when the check could not confirm it. It is never absent, so a caller
cannot mistake "not checked" for "fine".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.ai.execution.tools import run_tool_async


@dataclass(frozen=True)
class Verification:
    checked: bool
    confirmed: bool
    detail: str


def _values(node: Any, depth: int = 0) -> list[Any]:
    """Every scalar inside a decoded JSON value, so a check does not depend on shape."""

    if depth > 6:
        return []
    if isinstance(node, dict):
        out: list[Any] = []
        for value in node.values():
            out.extend(_values(value, depth + 1))
        return out
    if isinstance(node, list):
        out = []
        for item in node:
            out.extend(_values(item, depth + 1))
        return out
    return [node]


def _contains(decoded: Any, wanted: str) -> bool:
    target = str(wanted).strip().casefold()
    if not target:
        return False
    return any(str(value).strip().casefold() == target for value in _values(decoded))


def _items(decoded: Any) -> list[dict[str, Any]]:
    """The dicts inside a decoded value, whatever the wrapper is called."""

    if isinstance(decoded, dict):
        for value in decoded.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]
        return [decoded]
    if isinstance(decoded, list):
        return [item for item in decoded if isinstance(item, dict)]
    return []


async def _read(user_id: str, tool: str, payload: dict[str, Any]) -> Any | None:
    """Run a read tool and decode it, or None when it cannot answer."""

    try:
        result = await run_tool_async(
            tool,
            json.dumps({**payload, "user_id": user_id}, ensure_ascii=False, separators=(",", ":")),
            user_id=user_id,
        )
    except Exception:
        return None
    if not getattr(result, "safe", False):
        return None
    try:
        return json.loads(result.output)
    except (TypeError, ValueError):
        return None


def _unconfirmed(detail: str) -> Verification:
    return Verification(checked=True, confirmed=False, detail=detail)


async def verify_write(
    user_id: str,
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> Verification:
    """Read the state back and say whether the write is actually there."""

    repository = str(arguments.get("repository") or "").strip()

    if tool in {"github_create_branch", "github_delete_branch"}:
        branch = str(arguments.get("branch") or "").strip()
        decoded = await _read(user_id, "github_list_branches", {"repository": repository, "per_page": 100})
        if decoded is None:
            return _unconfirmed("I could not read the branch list back, so I cannot confirm it.")
        present = _contains(decoded, branch)
        if tool == "github_create_branch":
            return (
                Verification(True, True, f"branch {branch} is in the repository's branch list.")
                if present
                else _unconfirmed(f"the branch list does not contain {branch}.")
            )
        return (
            Verification(True, True, f"branch {branch} is no longer in the branch list.")
            if not present
            else _unconfirmed(f"branch {branch} is still in the branch list.")
        )

    if tool in {"github_create_pull_request", "github_merge_pull_request"}:
        number = arguments.get("number")
        head = str(arguments.get("head") or "").strip()
        decoded = await _read(user_id, "github_list_pull_requests", {"repository": repository, "per_page": 100})
        if decoded is None:
            return _unconfirmed("I could not read the pull request list back, so I cannot confirm it.")
        items = _items(decoded)
        match: dict[str, Any] | None = None
        for item in items:
            if number is not None and str(item.get("number")) == str(number):
                match = item
                break
            if head and str(item.get("head") or item.get("head_ref") or "").strip() == head:
                match = item
        if tool == "github_create_pull_request":
            if match is None:
                return _unconfirmed("the pull request list does not show it.")
            return Verification(True, True, "the pull request is in the repository's list.")
        if match is None:
            return _unconfirmed("I could not find the pull request to check whether it merged.")
        state = " ".join(str(value) for value in _values(match)).casefold()
        if "merged" in state or match.get("merged_at"):
            return Verification(True, True, "the pull request shows as merged.")
        return _unconfirmed("the pull request is still open, so it has not merged.")

    if tool == "github_update_repository_file":
        path = str(arguments.get("path") or "").strip()
        content = str(arguments.get("content") or "")
        decoded = await _read(
            user_id,
            "github_read_repository_file",
            {"repository": repository, "path": path, "ref": str(arguments.get("branch") or "")},
        )
        if decoded is None:
            return _unconfirmed("I could not read the file back, so I cannot confirm the commit landed.")
        seen = " ".join(str(value) for value in _values(decoded))
        marker = " ".join(content.split())[:40]
        if marker and marker in " ".join(seen.split()):
            return Verification(True, True, f"the file now contains what was written to {path}.")
        return _unconfirmed(f"the file at {path} does not show the new content yet.")

    return Verification(checked=False, confirmed=False, detail="No read-back is defined for this tool.")
