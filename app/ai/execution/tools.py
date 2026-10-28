from __future__ import annotations

import ast
import asyncio
import inspect
import json
import operator
import re
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.ai.agents.coding import code_analysis_tool, code_fix_suggestions_tool, code_transform_tool
from datetime import datetime, timedelta, timezone

from app.ai.gateway.observability import track
from app.ai.gateway.ai_rules import CONFIRM_OR_CANCEL
from app.ai.execution import action_chips
from app.ai.execution.approval import issue_approval_token, validate_approval_token
from app.ai.agents.code_sandbox import sandbox_execution_tool
from app.ai.execution.connector_authorization import authorize_tool
from app.ai.execution.tool_registry import get_tool_spec
from app.plugins.github.service import (
    get_github_commit as fetch_github_commit,
    list_github_branches as fetch_github_branches,
    list_github_check_runs as fetch_github_check_runs,
    list_github_pull_request_files as fetch_github_pull_request_files,
    list_github_issues as fetch_github_issues,
    list_github_pull_requests as fetch_github_pull_requests,
    list_github_repositories as fetch_github_repositories,
    list_github_repository_contents as fetch_github_repository_contents,
    read_github_repository_file as fetch_github_repository_file,
    search_github_repository_code as fetch_github_repository_code,
    list_github_repository_commits as fetch_github_repository_commits,
    compare_github_repository_commits as fetch_github_repository_compare,
)
from app.plugins.github.writes import (
    github_create_branch as write_github_create_branch,
    github_create_pull_request as write_github_create_pull_request,
    github_delete_branch as write_github_delete_branch,
    github_merge_pull_request as write_github_merge_pull_request,
    github_update_repository_file as commit_github_repository_file,
)
from app.channels.instagram.capabilities.instagram import (
    get_profile as get_instagram_profile,
    list_media as list_instagram_media,
)
from app.channels.instagram.capabilities.instagram_advanced import get_media_details as get_instagram_media_details
from app.capabilities.media.data_analysis import analyze_payload
from app.capabilities.platform.task_scheduler import build_schedule as build_task_schedule

MAX_CALCULATOR_ABS_VALUE = 10**100
MAX_CALCULATOR_EXPONENT = 1000
MAX_TOOL_PAYLOAD = 100_000
MAX_TOOL_TIMEOUT_SECONDS = 60.0
MAX_TOOL_RUN_SECONDS = 60.0
MAX_SYNC_TOOL_WORKERS = 8

ToolCallable = Callable[[str], str | Awaitable[str]]


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str
    safe: bool = True
    retryable: bool = False
    truncated: bool = False


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


async def _run_async(coro, timeout_seconds: float):
    """Await a coroutine with a bounded timeout on the current event loop."""
    bounded_timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    return await asyncio.wait_for(coro, timeout=bounded_timeout)


_SYNC_TOOL_SEMAPHORE = threading.BoundedSemaphore(MAX_SYNC_TOOL_WORKERS)


def _run_sync_with_timeout(tool: ToolCallable, payload: str, timeout_seconds: float) -> str:
    bounded_timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    if not _SYNC_TOOL_SEMAPHORE.acquire(blocking=False):
        raise RuntimeError("too many timed-out or running sync tools")

    result: list[object] = []
    errors: list[BaseException] = []
    release_lock = threading.Lock()
    released = {"value": False}

    def release_once() -> None:
        # A worker that outlives its timeout is abandoned, but its slot must still be
        # returned - exactly once - or eight hung tools would exhaust the pool for
        # the life of the process.
        with release_lock:
            if released["value"]:
                return
            released["value"] = True
        _SYNC_TOOL_SEMAPHORE.release()

    def runner() -> None:
        try:
            output = tool(payload)
            if inspect.isawaitable(output):
                output = asyncio.run(asyncio.wait_for(output, timeout=bounded_timeout))
            result.append(output)
        except BaseException as exc:
            errors.append(exc)
        finally:
            release_once()

    thread = threading.Thread(target=runner, daemon=True, name="indoone-sync-tool")
    thread.start()
    thread.join(bounded_timeout)
    if thread.is_alive():
        release_once()
        raise TimeoutError("tool execution timed out")
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("tool execution returned no result")
    output = result[0]
    if not isinstance(output, str):
        raise TypeError("tool output must be text")
    return output


async def _github_list_repositories(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_repositories payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_list_repositories")
    result = await _run_async(
        fetch_github_repositories(
            user_id=user_id,
            page=int(data.get("page", 1)),
            per_page=max(1, min(int(data.get("per_page", 10)), 100)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_assigned_issues(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_assigned_issues payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_list_assigned_issues")
    result = await _run_async(
        fetch_github_issues(
            user_id=user_id,
            page=int(data.get("page", 1)),
            per_page=max(1, min(int(data.get("per_page", 10)), 100)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_pull_requests(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_pull_requests payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not repository:
        raise ValueError("repository owner/name is required")
    spec = get_tool_spec("github_list_pull_requests")
    result = await _run_async(
        fetch_github_pull_requests(
            user_id=user_id,
            repository=repository,
            page=int(data.get("page", 1)),
            per_page=max(1, min(int(data.get("per_page", 10)), 100)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_repository_contents(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_repository_contents payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    if not user_id or not repository:
        raise ValueError("user_id and repository owner/name are required")
    spec = get_tool_spec("github_list_repository_contents")
    result = await _run_async(
        fetch_github_repository_contents(
            user_id, repository, str(data.get("path", "")), str(data.get("ref", ""))
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_read_repository_file(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_read_repository_file payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    path = str(data.get("path", "")).strip()
    if not user_id or not repository or not path:
        raise ValueError("user_id, repository and file path are required")
    spec = get_tool_spec("github_read_repository_file")
    result = await _run_async(
        fetch_github_repository_file(user_id, repository, path, str(data.get("ref", ""))),
        spec.timeout_seconds if spec else 30.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_search_repository_code(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_search_repository_code payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    query = str(data.get("query", "")).strip()
    if not user_id or not repository or not query:
        raise ValueError("user_id, repository and search query are required")
    spec = get_tool_spec("github_search_repository_code")
    result = await _run_async(
        fetch_github_repository_code(
            user_id, repository, query, max(1, min(int(data.get("per_page", 10)), 30))
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_repository_commits(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_repository_commits payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    if not user_id or not repository:
        raise ValueError("user_id and repository owner/name are required")
    spec = get_tool_spec("github_list_repository_commits")
    result = await _run_async(
        fetch_github_repository_commits(
            user_id, repository, str(data.get("path", "")),
            max(1, min(int(data.get("per_page", 10)), 30)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_compare_repository_commits(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_compare_repository_commits payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    base = str(data.get("base", "")).strip()
    head = str(data.get("head", "")).strip()
    if not user_id or not repository or not base or not head:
        raise ValueError("user_id, repository, base and head references are required")
    spec = get_tool_spec("github_compare_repository_commits")
    result = await _run_async(
        fetch_github_repository_compare(user_id, repository, base, head),
        spec.timeout_seconds if spec else 30.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_update_repository_file(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_update_repository_file payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    repository = str(data.get("repository", "")).strip()
    path = str(data.get("path", "")).strip()
    content = data.get("content")
    message = str(data.get("message", "")).strip()
    expected_sha = str(data.get("sha", "")).strip()
    create = bool(data.get("create", False))
    if not user_id or not repository or not path or not isinstance(content, str) or not message:
        raise ValueError("user_id, repository, path, content and commit message are required")
    if not create and not expected_sha:
        raise ValueError("the current file SHA is required when updating an existing file")

    # The target branch decides whether this is a reversible edit or a write to
    # something that other people and the deploy depend on. A protected branch needs a
    # confirmation even though the edit itself can be undone.
    from app.plugins.github.branch_protection import protected_branch_reason

    protected_reason = await protected_branch_reason(user_id, repository, str(data.get("branch", "")))
    held = _hold_write(
        data, user_id, "github_update_repository_file", protected_reason=protected_reason
    )
    if held is not None:
        return held
    # Past the hold, the guard above has already decided this write may run: either a
    # valid token arrived, or the branch is not protected and the edit is reversible.
    # The old unconditional token check here refused the reversible case and reached the
    # user as a failed tool with nothing to confirm - it was the one write whose
    # executor disagreed with its own spec about whether approval was required.
    spec = get_tool_spec("github_update_repository_file")
    result = await _run_async(
        commit_github_repository_file(
            user_id=user_id,
            repository=repository,
            path=path,
            content=content,
            message=message,
            expected_sha=expected_sha,
            approval_verified=True,
            branch=str(data.get("branch", "")),
            create=create,
        ),
        spec.timeout_seconds if spec else 30.0,
    )
    return await _verified_write_result(user_id, "github_update_repository_file", data, result)


async def _instagram_get_profile(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("instagram_get_profile")
    result = await _run_async(get_instagram_profile(user_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _instagram_list_media(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("instagram_list_media")
    result = await _run_async(
        list_instagram_media(
            user_id,
            limit=max(1, min(int(data.get("limit", 25)), 100)),
            after=str(data.get("after", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _instagram_get_media_details(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("instagram_get_media_details payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    media_id = str(data.get("media_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not media_id:
        raise ValueError("media_id is required")
    spec = get_tool_spec("instagram_get_media_details")
    result = await _run_async(
        get_instagram_media_details(user_id, media_id), spec.timeout_seconds if spec else 20.0
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _data_analysis(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("data_analysis payload must be an object")
    content = str(data.get("content", ""))
    if not content.strip():
        raise ValueError("content is required")
    filename = str(data.get("filename", "data.csv")).strip() or "data.csv"
    result = analyze_payload(filename, content.encode("utf-8"))
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _task_schedule_build(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("task_schedule_build payload must be an object")
    schedule = str(data.get("schedule", "")).strip()
    if not schedule:
        raise ValueError("schedule is required")
    decision = build_task_schedule(
        schedule,
        timezone=str(data.get("timezone", "UTC")) or "UTC",
        retry_limit=int(data.get("retry_limit", 3)),
        misfire_policy=str(data.get("misfire_policy", "skip")) or "skip",
    )
    return json.dumps(
        {
            "enabled": decision.enabled,
            "schedule": decision.schedule,
            "timezone": decision.timezone,
            "misfire_policy": decision.misfire_policy,
            "retry_limit": decision.retry_limit,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


async def _github_list_branches(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_branches payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_list_branches")
    result = await _run_async(
        fetch_github_branches(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            per_page=max(1, min(int(data.get("per_page", 30)), 100)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_pull_request_files(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_pull_request_files payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    number = int(data.get("number", 0))
    if number < 1:
        raise ValueError("pull request number is required")
    spec = get_tool_spec("github_list_pull_request_files")
    result = await _run_async(
        fetch_github_pull_request_files(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            number=number,
            per_page=max(1, min(int(data.get("per_page", 30)), 100)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_list_check_runs(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_list_check_runs payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_list_check_runs")
    result = await _run_async(
        fetch_github_check_runs(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            ref=str(data.get("ref", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_get_commit(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_get_commit payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    sha = str(data.get("sha", "")).strip()
    if not sha:
        raise ValueError("commit sha is required")
    spec = get_tool_spec("github_get_commit")
    result = await _run_async(
        fetch_github_commit(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            sha=sha,
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _approved_write(data: dict[str, object], user_id: str, tool: str) -> bool:
    """A write runs only with a server-issued approval token for this exact tool.

    The model proposes; the user confirms; the backend issues the token. Without
    it the tool refuses, so a hallucinated call cannot change a repository.
    """
    token = str(data.get("approval_token", "")).strip()
    return bool(validate_approval_token(token, user_id, tool))


# The wording the user is asked to confirm, one per write.
_WRITE_INTENT = {
    "github_update_repository_file": "commit that file change",
    "github_create_branch": "create a new branch",
    "github_delete_branch": "delete the branch",
    "github_create_pull_request": "open a pull request",
    "github_merge_pull_request": "merge the pull request",
}
_PENDING_CONVERSATION = "github-pending"

# The wording a user uses when they ask for a write themselves. An explicit
# request in the user's own message IS the approval - the assistant acts on it
# instead of asking the user to confirm the thing they just asked for. Only the
# user's message is read here, never the model's wording.
_WRITE_REQUEST_PATTERNS = {
    "github_create_branch": r"\b(?:create|make|add|open|new)\b[^.\n]{0,40}\bbranch\b|\bbranch\b[^.\n]{0,40}\b(?:create|madi|maadu|ಮಾಡು|ಮಾಡಿ)\b",
    "github_delete_branch": r"\b(?:delete|remove|drop)\b[^.\n]{0,40}\bbranch\b",
    "github_create_pull_request": r"\b(?:create|open|raise|make|file)\b[^.\n]{0,40}\b(?:pull\s*request|pr)\b",
    "github_merge_pull_request": r"\bmerge\b[^.\n]{0,40}\b(?:pull\s*request|pr|branch)\b|\b(?:pull\s*request|pr)\b[^.\n]{0,40}\bmerge\b",
}


def _explicit_write_intent(tool: str, message: str) -> bool:
    """Whether the user's own message asks for this exact write."""
    pattern = _WRITE_REQUEST_PATTERNS.get(tool)
    if not pattern or not message.strip():
        return False
    return bool(re.search(pattern, message, re.IGNORECASE))


def _write_approved(
    data: dict[str, object], user_id: str, tool: str, *, protected_reason: str = ""
) -> bool:
    """Whether this write may run now.

    Three ways, in order: the tool's own policy says it runs when asked (creating a
    branch, opening a pull request, writing a file), a server-issued token arrived, or
    the user's own message asked for it. Deleting a branch and merging a pull request
    are not in the first group - they still need the token.

    ``protected_reason`` is the fourth case and it overrides the first and the third:
    when the target branch is protected, neither "this tool is reversible" nor "the
    user asked for it" is enough. Only a token is. Reversibility says the change can be
    undone; it says nothing about what breaks while it is applied.
    """
    if _approved_write(data, user_id, tool):
        return True
    if protected_reason:
        return False
    spec = get_tool_spec(tool)
    if spec is not None and not spec.requires_approval:
        return True
    return _explicit_write_intent(tool, str(data.get("original_message") or ""))


def _hold_write(
    data: dict[str, object], user_id: str, tool: str, *, protected_reason: str = ""
) -> str | None:
    """Hold an unapproved write and return the proposal, or None to let it run.

    This lives in the executor because the executor is the one place a write is
    guaranteed to pass, whichever path reached it. Putting it in the GitHub
    handler was the earlier mistake: the handler returns before the model runs, so
    a write never got there, nothing was stored, and no token was ever issued.
    """
    if _write_approved(data, user_id, tool, protected_reason=protected_reason):
        return None
    try:
        token = issue_approval_token(user_id, tool, ttl_seconds=300)
        from app.capabilities.connectors.integrations import _fernet
        from app.capabilities.connectors.store import save_pending_github_action

        stored = {
            key: value for key, value in data.items()
            if key not in {"user_id", "approval_token", "conversation_id", "original_message"}
        }
        payload = _fernet().encrypt(
            json.dumps({"tool": tool, "arguments": stored, "token": token}, separators=(",", ":")).encode("utf-8")
        )
        conversation = str(data.get("conversation_id") or "").strip() or _PENDING_CONVERSATION
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()
        save_pending_github_action(user_id, conversation, payload, expires_at)
    except (RuntimeError, ValueError, ImportError):
        return json.dumps({
            "ok": False,
            "code": "APPROVAL_UNAVAILABLE",
            "reply": (
                "GitHub changes cannot be confirmed on this deployment, so nothing was changed. "
                "The approval secret is not configured."
            ),
        }, ensure_ascii=False, sort_keys=True)
    # Written for the USER, because this text is the reply now - not a note to a
    # model. Nothing has happened yet and it says so.
    reason = protected_reason.strip()
    # The same two answers the sentence asks for, as data: the client renders them as
    # buttons, and pressing one sends exactly the word the user would have typed.
    action_chips.emit(action_chips.APPROVE, action_chips.CANCEL)
    return json.dumps({
        "ok": False,
        "code": "APPROVAL_REQUIRED",
        "reply": (
            f"I will {_WRITE_INTENT.get(tool, 'make this change')} in "
            f"{data.get('repository') or 'the repository'}. "
            + (f"{reason} " if reason else "")
            + CONFIRM_OR_CANCEL
        ),
    }, ensure_ascii=False, sort_keys=True)


async def _verified_write_result(user_id: str, tool: str, data: dict[str, object], result: object) -> str:
    """A write's result, with the state read back before anything says it worked.

    The write returning ok only means the provider accepted the call. The reply must be
    about what is actually there, so every write is followed by an ordinary read and the
    result carries what that read saw.
    """
    from app.plugins.github.verify import verify_write

    payload = dict(result) if isinstance(result, dict) else {"result": result}
    verification = await verify_write(user_id, tool, data, payload)
    payload["verified"] = verification.confirmed
    payload["checked"] = verification.checked
    payload["verification"] = verification.detail
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _github_create_branch(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_create_branch payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_create_branch")
    held = _hold_write(data, user_id, "github_create_branch")
    if held is not None:
        return held
    result = await _run_async(
        write_github_create_branch(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            branch=str(data.get("branch", "")),
            from_ref=str(data.get("from_ref", "")),
            approved=_write_approved(data, user_id, "github_create_branch"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return await _verified_write_result(user_id, "github_create_branch", data, result)


async def _github_delete_branch(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_delete_branch payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_delete_branch")
    held = _hold_write(data, user_id, "github_delete_branch")
    if held is not None:
        return held
    result = await _run_async(
        write_github_delete_branch(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            branch=str(data.get("branch", "")),
            approved=_write_approved(data, user_id, "github_delete_branch"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return await _verified_write_result(user_id, "github_delete_branch", data, result)


async def _github_create_pull_request(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_create_pull_request payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_create_pull_request")
    held = _hold_write(data, user_id, "github_create_pull_request")
    if held is not None:
        return held
    result = await _run_async(
        write_github_create_pull_request(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            title=str(data.get("title", "")),
            head=str(data.get("head", "")),
            base=str(data.get("base", "main")) or "main",
            body=str(data.get("body", "")),
            approved=_write_approved(data, user_id, "github_create_pull_request"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return await _verified_write_result(user_id, "github_create_pull_request", data, result)


async def _github_merge_pull_request(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_merge_pull_request payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    number = int(data.get("number", 0))
    if number < 1:
        raise ValueError("pull request number is required")
    spec = get_tool_spec("github_merge_pull_request")
    held = _hold_write(data, user_id, "github_merge_pull_request")
    if held is not None:
        return held
    result = await _run_async(
        write_github_merge_pull_request(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            number=number,
            method=str(data.get("method", "squash")) or "squash",
            approved=_write_approved(data, user_id, "github_merge_pull_request"),
        ),
        spec.timeout_seconds if spec else 30.0,
    )
    return await _verified_write_result(user_id, "github_merge_pull_request", data, result)


def _find_tools(query: str) -> str:
    """The find_tools executor: the searchable catalogue, as text.

    The tool loop calls the same builder directly, because it needs the names to
    offer on the next round; this entry exists so the registry contract holds - every
    declared tool has a runtime implementation - and so the agent path can search too.
    """
    from app.ai.execution.tool_search import find_tools_payload

    return json.dumps(find_tools_payload(query), ensure_ascii=False, sort_keys=True)


TOOLS: dict[str, ToolCallable] = {
    "find_tools": _find_tools,
    "calculator": _calculate,
    "text_stats": _text_stats,
    "json_summary": _json_summary,
    "code_analysis": code_analysis_tool,
    "code_fix_suggestions": code_fix_suggestions_tool,
    "code_transform": code_transform_tool,
    "sandbox_execution": sandbox_execution_tool,
    "github_list_repositories": _github_list_repositories,
    "github_list_branches": _github_list_branches,
    "github_list_pull_request_files": _github_list_pull_request_files,
    "github_list_check_runs": _github_list_check_runs,
    "github_get_commit": _github_get_commit,
    "github_create_branch": _github_create_branch,
    "github_delete_branch": _github_delete_branch,
    "github_create_pull_request": _github_create_pull_request,
    "github_merge_pull_request": _github_merge_pull_request,
    "github_list_assigned_issues": _github_list_assigned_issues,
    "github_list_pull_requests": _github_list_pull_requests,
    "github_list_repository_contents": _github_list_repository_contents,
    "github_read_repository_file": _github_read_repository_file,
    "github_search_repository_code": _github_search_repository_code,
    "github_list_repository_commits": _github_list_repository_commits,
    "github_compare_repository_commits": _github_compare_repository_commits,
    "github_update_repository_file": _github_update_repository_file,
    "instagram_get_profile": _instagram_get_profile,
    "instagram_list_media": _instagram_list_media,
    "instagram_get_media_details": _instagram_get_media_details,
    "data_analysis": _data_analysis,
    "task_schedule_build": _task_schedule_build,
}


def _validate_tool_request(name: str, payload: str) -> tuple[str, object, str | None]:
    normalized_name = str(name).strip().lower()
    if not isinstance(payload, str):
        return normalized_name, None, "Tool payload must be text"
    if len(payload) > MAX_TOOL_PAYLOAD:
        return normalized_name, None, "Tool payload is too large"
    spec = get_tool_spec(normalized_name)
    tool = TOOLS.get(normalized_name)
    if spec is None or tool is None:
        return normalized_name, None, "Unknown or unregistered tool"
    return normalized_name, (spec, tool), None


def _authorize_execution(user_id: str, spec: object) -> str | None:
    connector_id = getattr(spec, "connector_id", None)
    if not connector_id:
        return None
    if not user_id.strip():
        return "authenticated user is required for connector tools"
    capability = getattr(spec, "capability", None)
    if not capability:
        return "tool connector metadata is incomplete"
    decision = authorize_tool(user_id, getattr(spec, "name", ""))
    return None if decision.allowed else decision.reason


def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError))


def _tool_error_message(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "Tool error: tool execution timed out"
    message = str(exc).strip()
    return f"Tool error: {message}" if message else f"Tool error: {type(exc).__name__}"


async def run_tool_async(name: str, payload: str, max_runtime_seconds: float | None = None, user_id: str = "") -> ToolResult:
    """Run a tool from an async request, measured as one span.

    The measurement wraps the call and changes nothing about it: the same
    validation, the same authorization, the same timeout and the same result. Only
    the name, the outcome and the duration are recorded, never the payload or the
    output.
    """

    async with track("tool.call", "tool", tool=str(name or "").strip().casefold()) as span:
        result = await _execute_tool_async(name, payload, max_runtime_seconds, user_id)
        span.attributes["ok"] = bool(result.safe)
        span.attributes["truncated"] = bool(getattr(result, "truncated", False))
        if not result.safe:
            span.fail("tool_error")
        return result


async def _execute_tool_async(name: str, payload: str, max_runtime_seconds: float | None = None, user_id: str = "") -> ToolResult:
    """Run a tool from an async request without creating a thread for async tools."""
    normalized_name, prepared, error = _validate_tool_request(name, payload)
    if error is not None:
        return ToolResult(normalized_name, error, safe=False, retryable=False)
    spec, tool = prepared
    authorization_error = _authorize_execution(user_id, spec)
    if authorization_error is not None:
        return ToolResult(normalized_name, f"Tool error: {authorization_error}", safe=False, retryable=False)
    started = time.monotonic()
    requested_budget = MAX_TOOL_RUN_SECONDS if max_runtime_seconds is None else float(max_runtime_seconds)
    total_budget = max(0.01, min(requested_budget, MAX_TOOL_RUN_SECONDS, MAX_TOOL_TIMEOUT_SECONDS))
    try:
        remaining = total_budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("tool run budget exceeded")
        timeout_seconds = min(float(spec.timeout_seconds), remaining)
        if inspect.iscoroutinefunction(tool):
            output = await _run_async(tool(payload), timeout_seconds)
        else:
            # A sync tool runs once, in a worker thread. Calling it here first would
            # run it a second time on the event loop with no timeout, doubling the
            # work and stalling every other request while it runs.
            output = await asyncio.wait_for(
                asyncio.to_thread(_run_sync_with_timeout, tool, payload, timeout_seconds),
                timeout=timeout_seconds,
            )
        if time.monotonic() - started > total_budget:
            raise TimeoutError("tool run budget exceeded")
        if not isinstance(output, str):
            raise TypeError("tool output must be text")
        truncated = len(output) > spec.max_output_chars
        if truncated:
            output = output[: spec.max_output_chars] + "\n[output truncated by policy]"
        return ToolResult(normalized_name, output, safe=True, retryable=False, truncated=truncated)
    except Exception as exc:
        return ToolResult(normalized_name, _tool_error_message(exc), safe=False, retryable=_is_retryable_exception(exc))


def run_tool(name: str, payload: str, max_runtime_seconds: float | None = None, user_id: str = "") -> ToolResult:
    """Run a tool on the calling thread, measured as one span.

    This is the synchronous entry, used by the agent from a worker thread. The span
    still reaches the aggregate metrics, but it cannot attach to the request trace,
    because a new thread does not inherit the request context.
    """

    with track("tool.call", "tool", tool=str(name or "").strip().casefold()) as span:
        result = _execute_tool(name, payload, max_runtime_seconds, user_id)
        span.attributes["ok"] = bool(result.safe)
        span.attributes["truncated"] = bool(getattr(result, "truncated", False))
        if not result.safe:
            span.fail("tool_error")
        return result


def _execute_tool(name: str, payload: str, max_runtime_seconds: float | None = None, user_id: str = "") -> ToolResult:
    normalized_name, prepared, error = _validate_tool_request(name, payload)
    if error is not None:
        return ToolResult(normalized_name, error, safe=False, retryable=False)
    spec, tool = prepared
    authorization_error = _authorize_execution(user_id, spec)
    if authorization_error is not None:
        return ToolResult(normalized_name, f"Tool error: {authorization_error}", safe=False, retryable=False)
    started = time.monotonic()
    requested_budget = MAX_TOOL_RUN_SECONDS if max_runtime_seconds is None else float(max_runtime_seconds)
    total_budget = max(0.01, min(requested_budget, MAX_TOOL_RUN_SECONDS, MAX_TOOL_TIMEOUT_SECONDS))
    try:
        elapsed = time.monotonic() - started
        remaining = total_budget - elapsed
        if remaining <= 0:
            raise TimeoutError("tool run budget exceeded")
        timeout_seconds = min(float(spec.timeout_seconds), remaining)
        output = _run_sync_with_timeout(tool, payload, timeout_seconds)
        if time.monotonic() - started > total_budget:
            raise TimeoutError("tool run budget exceeded")
        truncated = len(output) > spec.max_output_chars
        if truncated:
            output = output[: spec.max_output_chars] + "\n[output truncated by policy]"
        return ToolResult(normalized_name, output, safe=True, retryable=False, truncated=truncated)
    except Exception as exc:
        return ToolResult(normalized_name, _tool_error_message(exc), safe=False, retryable=_is_retryable_exception(exc))
