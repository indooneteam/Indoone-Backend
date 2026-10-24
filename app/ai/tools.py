from __future__ import annotations

import ast
import asyncio
import inspect
import json
import operator
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.ai.coding import code_analysis_tool, code_fix_suggestions_tool, code_transform_tool
from app.ai.approval import validate_approval_token
from app.ai.code_sandbox import sandbox_execution_tool
from app.ai.connector_authorization import authorize_tool
from app.ai.tool_registry import get_tool_spec
from app.capabilities.contacts import parse_contacts, resolve_contact, search_contacts
from app.plugins.gmail.service import get_gmail_message, list_gmail_messages, send_gmail_message
from app.capabilities.email_safety import analyze_gmail_email_safety
from app.capabilities.email_actions import compose_gmail_email
from app.plugins.gmail.service import reply_gmail_message
from app.capabilities.phone import build_call_action
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
from app.capabilities.google_calendar import (
    create_event as create_google_calendar_event,
    delete_event as delete_google_calendar_event,
    get_event as get_google_calendar_event,
    list_calendars as list_google_calendars,
    list_events as list_google_calendar_events,
    update_event as update_google_calendar_event,
)
from app.capabilities.google_drive import (
    get_drive_file as get_google_drive_file,
    list_drive_files as list_google_drive_files,
    upload_drive_file as upload_google_drive_file,
)
from app.capabilities.google_photos import (
    get_media_item as get_google_photos_media_item,
    list_albums as list_google_photos_albums,
    list_media as list_google_photos_media,
    search_media as search_google_photos_media,
)
from app.capabilities.youtube import (
    get_my_channel as get_youtube_channel,
    get_videos as get_youtube_videos,
    list_my_videos as list_youtube_my_videos,
    search_youtube,
)
from app.instagram.capabilities.instagram import (
    get_profile as get_instagram_profile,
    list_media as list_instagram_media,
)
from app.instagram.capabilities.instagram_advanced import get_media_details as get_instagram_media_details
from app.capabilities.facebook import (
    list_page_posts as list_facebook_page_posts,
    list_pages as list_facebook_pages,
)
from app.capabilities.canva import (
    create_export_job as create_canva_export_job,
    get_design as get_canva_design,
    get_export_formats as get_canva_export_formats,
    list_designs as list_canva_designs,
)
from app.capabilities.data_analysis import analyze_payload
from app.capabilities.task_scheduler import build_schedule as build_task_schedule

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

    def runner() -> None:
        try:
            output = tool(payload)
            if inspect.isawaitable(output):
                output = asyncio.run(asyncio.wait_for(output, timeout=bounded_timeout))
            result.append(output)
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
    approval_token = str(data.get("approval_token", "")).strip()
    if not user_id or not repository or not path or not isinstance(content, str) or not message:
        raise ValueError("user_id, repository, path, content and commit message are required")
    if not create and not expected_sha:
        raise ValueError("the current file SHA is required when updating an existing file")
    if not validate_approval_token(approval_token, user_id, "github_update_repository_file"):
        raise PermissionError("a valid, current, server-issued GitHub file-update approval is required")
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
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _gmail_search(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_search payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("gmail_search")
    result = await _run_async(list_gmail_messages(user_id=user_id, query=str(data.get("query", "")), page_token=str(data.get("page_token", "")), max_results=int(data.get("max_results", 20))), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _gmail_read(payload: str) -> str:
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
    result = await _run_async(get_gmail_message(user_id=user_id, message_id=message_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _gmail_compose(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_compose payload must be an object")
    request = str(data.get("request", "")).strip()
    if not request:
        raise ValueError("compose request is required")
    result = await _run_async(compose_gmail_email(request=request), 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _gmail_email_safety(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_email_safety payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("gmail_email_safety")
    return await _run_async(
        analyze_gmail_email_safety(
            user_id=user_id,
            query=str(data.get("query", "")),
            message_id=str(data.get("message_id", "")),
            user_request=str(data.get("user_request", "")),
            limit=int(data.get("limit", 1)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )


async def _gmail_reply(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_reply payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    message_id = str(data.get("message_id", "")).strip()
    body = str(data.get("body", "")).strip()
    if not user_id or not message_id or not body:
        raise ValueError("user_id, message_id and body are required")
    spec = get_tool_spec("gmail_reply")
    result = await _run_async(
        reply_gmail_message(
            user_id=user_id,
            message_id=message_id,
            body=body,
            approved=True,
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _gmail_send(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("gmail_send payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("gmail_send")
    result = await _run_async(
        send_gmail_message(
            user_id=user_id,
            to=str(data.get("to", "")),
            subject=str(data.get("subject", "")),
            body=str(data.get("body", "")),
            approved=True,
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_list_calendars(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_list_calendars payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_calendar_list_calendars")
    result = await _run_async(
        list_google_calendars(
            user_id,
            page_token=str(data.get("page_token", "")),
            max_results=max(1, min(int(data.get("max_results", 100)), 250)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_list_events(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_list_events payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_calendar_list_events")
    result = await _run_async(
        list_google_calendar_events(
            user_id,
            calendar_id=str(data.get("calendar_id", "primary")) or "primary",
            time_min=str(data.get("time_min", "")),
            time_max=str(data.get("time_max", "")),
            query=str(data.get("query", "")),
            page_token=str(data.get("page_token", "")),
            max_results=max(1, min(int(data.get("max_results", 100)), 250)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_get_event(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_get_event payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    event_id = str(data.get("event_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not event_id:
        raise ValueError("event_id is required")
    spec = get_tool_spec("google_calendar_get_event")
    result = await _run_async(
        get_google_calendar_event(user_id, str(data.get("calendar_id", "primary")) or "primary", event_id),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_create_event(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_create_event payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    event = data.get("event")
    if not user_id:
        raise ValueError("user_id is required")
    if not isinstance(event, dict):
        raise ValueError("event is required")
    spec = get_tool_spec("google_calendar_create_event")
    result = await _run_async(
        create_google_calendar_event(user_id, str(data.get("calendar_id", "primary")) or "primary", event, approved=True),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_update_event(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_update_event payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    event_id = str(data.get("event_id", "")).strip()
    event = data.get("event")
    if not user_id:
        raise ValueError("user_id is required")
    if not event_id:
        raise ValueError("event_id is required")
    if not isinstance(event, dict):
        raise ValueError("event is required")
    spec = get_tool_spec("google_calendar_update_event")
    result = await _run_async(
        update_google_calendar_event(user_id, str(data.get("calendar_id", "primary")) or "primary", event_id, event, approved=True),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_calendar_delete_event(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_calendar_delete_event payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    event_id = str(data.get("event_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not event_id:
        raise ValueError("event_id is required")
    spec = get_tool_spec("google_calendar_delete_event")
    result = await _run_async(
        delete_google_calendar_event(user_id, str(data.get("calendar_id", "primary")) or "primary", event_id, approved=True),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_drive_list_files(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_drive_list_files payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_drive_list_files")
    result = await _run_async(
        list_google_drive_files(
            user_id,
            query=str(data.get("query", "")),
            page_token=str(data.get("page_token", "")),
            page_size=max(1, min(int(data.get("page_size", 50)), 1000)),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_drive_get_file(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_drive_get_file payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    file_id = str(data.get("file_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not file_id:
        raise ValueError("file_id is required")
    spec = get_tool_spec("google_drive_get_file")
    result = await _run_async(
        get_google_drive_file(
            user_id,
            file_id,
            download=bool(data.get("download", False)),
            export_mime_type=str(data.get("export_mime_type", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_drive_upload_file(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_drive_upload_file payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    filename = str(data.get("filename", "")).strip()
    mime_type = str(data.get("mime_type", "")).strip()
    content = str(data.get("content", ""))
    if not user_id:
        raise ValueError("user_id is required")
    if not filename or not mime_type:
        raise ValueError("filename and mime_type are required")
    spec = get_tool_spec("google_drive_upload_file")
    result = await _run_async(
        upload_google_drive_file(
            user_id,
            filename,
            mime_type,
            content.encode("utf-8"),
            parent_id=str(data.get("parent_id", "")),
            approved=True,
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_photos_list_media(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_photos_list_media payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_photos_list_media")
    result = await _run_async(
        list_google_photos_media(
            user_id,
            page_size=max(1, min(int(data.get("page_size", 25)), 100)),
            page_token=str(data.get("page_token", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_photos_search_media(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_photos_search_media payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_photos_search_media")
    result = await _run_async(
        search_google_photos_media(
            user_id,
            page_size=max(1, min(int(data.get("page_size", 25)), 100)),
            page_token=str(data.get("page_token", "")),
            album_id=str(data.get("album_id", "")),
            media_type=str(data.get("media_type", "")),
            order=str(data.get("order", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_photos_get_media_item(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_photos_get_media_item payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    media_item_id = str(data.get("media_item_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not media_item_id:
        raise ValueError("media_item_id is required")
    spec = get_tool_spec("google_photos_get_media_item")
    result = await _run_async(
        get_google_photos_media_item(user_id, media_item_id),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _google_photos_list_albums(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("google_photos_list_albums payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("google_photos_list_albums")
    result = await _run_async(
        list_google_photos_albums(
            user_id,
            page_size=max(1, min(int(data.get("page_size", 20)), 50)),
            page_token=str(data.get("page_token", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _youtube_search(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("youtube_search payload must be an object")
    query = str(data.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    spec = get_tool_spec("youtube_search")
    result = await _run_async(
        search_youtube(
            query,
            resource_type=str(data.get("resource_type", "video")) or "video",
            max_results=max(1, min(int(data.get("max_results", 10)), 50)),
            page_token=str(data.get("page_token", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _youtube_get_videos(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("youtube_get_videos payload must be an object")
    ids = data.get("video_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("video_ids is required")
    user_id = str(data.get("user_id", "")).strip()
    spec = get_tool_spec("youtube_get_videos")
    result = await _run_async(
        get_youtube_videos([str(item) for item in ids], user_id=user_id),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _youtube_get_my_channel(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("youtube_get_my_channel")
    result = await _run_async(get_youtube_channel(user_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _youtube_list_my_videos(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("youtube_list_my_videos")
    result = await _run_async(
        list_youtube_my_videos(
            user_id,
            max_results=max(1, min(int(data.get("max_results", 20)), 50)),
            page_token=str(data.get("page_token", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


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


async def _facebook_list_pages(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("facebook_list_pages")
    result = await _run_async(list_facebook_pages(user_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _facebook_list_page_posts(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("facebook_list_page_posts payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    page_id = str(data.get("page_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not page_id:
        raise ValueError("page_id is required")
    spec = get_tool_spec("facebook_list_page_posts")
    result = await _run_async(
        list_facebook_page_posts(
            user_id,
            page_id,
            limit=max(1, min(int(data.get("limit", 25)), 100)),
            after=str(data.get("after", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _canva_list_designs(payload: str) -> str:
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        data = {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("canva_list_designs")
    result = await _run_async(
        list_canva_designs(
            user_id,
            query=str(data.get("query", "")),
            continuation=str(data.get("continuation", "")),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _canva_get_design(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("canva_get_design payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    design_id = str(data.get("design_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not design_id:
        raise ValueError("design_id is required")
    spec = get_tool_spec("canva_get_design")
    result = await _run_async(get_canva_design(user_id, design_id), spec.timeout_seconds if spec else 20.0)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _canva_get_export_formats(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("canva_get_export_formats payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    design_id = str(data.get("design_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    if not design_id:
        raise ValueError("design_id is required")
    spec = get_tool_spec("canva_get_export_formats")
    result = await _run_async(
        get_canva_export_formats(user_id, design_id), spec.timeout_seconds if spec else 20.0
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _canva_create_export_job(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("canva_create_export_job payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    design_id = str(data.get("design_id", "")).strip()
    format_payload = data.get("format")
    if not user_id:
        raise ValueError("user_id is required")
    if not design_id:
        raise ValueError("design_id is required")
    if not isinstance(format_payload, dict):
        raise ValueError("format is required")
    spec = get_tool_spec("canva_create_export_job")
    result = await _run_async(
        create_canva_export_job(user_id, design_id, format_payload, approved=True),
        spec.timeout_seconds if spec else 20.0,
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
    spec = get_tool_spec("github_list_pull_request_files")
    result = await _run_async(
        fetch_github_pull_request_files(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            number=int(data.get("number", 0)),
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
    spec = get_tool_spec("github_get_commit")
    result = await _run_async(
        fetch_github_commit(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            sha=str(data.get("sha", "")),
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


async def _github_create_branch(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_create_branch payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_create_branch")
    result = await _run_async(
        write_github_create_branch(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            branch=str(data.get("branch", "")),
            from_ref=str(data.get("from_ref", "")),
            approved=_approved_write(data, user_id, "github_create_branch"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_delete_branch(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_delete_branch payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_delete_branch")
    result = await _run_async(
        write_github_delete_branch(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            branch=str(data.get("branch", "")),
            approved=_approved_write(data, user_id, "github_delete_branch"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_create_pull_request(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_create_pull_request payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_create_pull_request")
    result = await _run_async(
        write_github_create_pull_request(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            title=str(data.get("title", "")),
            head=str(data.get("head", "")),
            base=str(data.get("base", "main")) or "main",
            body=str(data.get("body", "")),
            approved=_approved_write(data, user_id, "github_create_pull_request"),
        ),
        spec.timeout_seconds if spec else 20.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _github_merge_pull_request(payload: str) -> str:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("github_merge_pull_request payload must be an object")
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required")
    spec = get_tool_spec("github_merge_pull_request")
    result = await _run_async(
        write_github_merge_pull_request(
            user_id=user_id,
            repository=str(data.get("repository", "")),
            number=int(data.get("number", 0)),
            method=str(data.get("method", "squash")) or "squash",
            approved=_approved_write(data, user_id, "github_merge_pull_request"),
        ),
        spec.timeout_seconds if spec else 30.0,
    )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


TOOLS: dict[str, ToolCallable] = {
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
    "gmail_read": _gmail_read,
    "gmail_compose": _gmail_compose,
    "gmail_email_safety": _gmail_email_safety,
    "gmail_reply": _gmail_reply,
    "gmail_send": _gmail_send,
    "google_calendar_list_calendars": _google_calendar_list_calendars,
    "google_calendar_list_events": _google_calendar_list_events,
    "google_calendar_get_event": _google_calendar_get_event,
    "google_calendar_create_event": _google_calendar_create_event,
    "google_calendar_update_event": _google_calendar_update_event,
    "google_calendar_delete_event": _google_calendar_delete_event,
    "google_drive_list_files": _google_drive_list_files,
    "google_drive_get_file": _google_drive_get_file,
    "google_drive_upload_file": _google_drive_upload_file,
    "google_photos_list_media": _google_photos_list_media,
    "google_photos_search_media": _google_photos_search_media,
    "google_photos_get_media_item": _google_photos_get_media_item,
    "google_photos_list_albums": _google_photos_list_albums,
    "youtube_search": _youtube_search,
    "youtube_get_videos": _youtube_get_videos,
    "youtube_get_my_channel": _youtube_get_my_channel,
    "youtube_list_my_videos": _youtube_list_my_videos,
    "instagram_get_profile": _instagram_get_profile,
    "instagram_list_media": _instagram_list_media,
    "instagram_get_media_details": _instagram_get_media_details,
    "facebook_list_pages": _facebook_list_pages,
    "facebook_list_page_posts": _facebook_list_page_posts,
    "canva_list_designs": _canva_list_designs,
    "canva_get_design": _canva_get_design,
    "canva_get_export_formats": _canva_get_export_formats,
    "canva_create_export_job": _canva_create_export_job,
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
        output = tool(payload)
        if inspect.isawaitable(output):
            output = await _run_async(output, timeout_seconds)
        else:
            output = await asyncio.wait_for(asyncio.to_thread(_run_sync_with_timeout, tool, payload, timeout_seconds), timeout=timeout_seconds)
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
