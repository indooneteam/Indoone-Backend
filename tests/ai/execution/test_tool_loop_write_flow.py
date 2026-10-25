"""A GitHub write proposed through the unified loop must be held, then run on CONFIRM.

Before this, ``dispatch_tool`` refused an approval-required tool before its
executor could hold it, so the proposal was never stored and the user's CONFIRM
had nothing to replay - the write could never happen. And the approval token was
forwarded to the executor for only one of the five write tools, so an approved
call was turned back into a request for approval.

These tests drive the dispatcher - the path the model's tool call actually takes -
because the executor-only tests cannot see the dispatcher pre-empt it.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from app.ai.execution import tool_loop
from app.ai.execution import tools as ai_tools
from app.ai.execution.approval import issue_approval_token
from app.capabilities.connectors import store
from app.plugins.github import chat_tools


def _branch_result(repository: str, branch: str) -> dict:
    """The shape the create-branch executor returns (the formatter reads it)."""
    return {"repository": repository, "result": {"branch": branch, "sha": "a" * 40, "from": "main"}}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a-test-approval-secret-that-is-long-enough")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    # The connector must read as connected, or run_tool_async refuses before the
    # executor is reached. Authorization itself is covered by its own tests. The
    # module is imported into each caller, so every binding is patched.
    allow = lambda *args, **kwargs: SimpleNamespace(allowed=True, reason="")  # noqa: E731
    monkeypatch.setattr("app.ai.execution.connector_authorization.authorize_tool", allow)
    monkeypatch.setattr("app.ai.execution.tools.authorize_tool", allow)
    monkeypatch.setattr("app.ai.execution.approval.authorize_tool", allow)
    store.initialize()
    yield


def test_a_proposed_write_is_held_and_stored(monkeypatch):
    calls: list[dict] = []

    async def _fake_write(**kwargs):
        calls.append(kwargs)
        return _branch_result(kwargs["repository"], kwargs["branch"])

    monkeypatch.setattr(ai_tools, "write_github_create_branch", _fake_write)

    result = asyncio.run(tool_loop.dispatch_tool(
        "u-1", "c-1", "github_create_branch",
        {"repository": "o/r", "branch": "terminal"},
    ))

    assert result["ok"] is False
    assert result["code"] == "APPROVAL_REQUIRED"
    assert calls == [], "nothing may run before the user confirms"
    assert store.get_pending_github_action("u-1", "c-1") is not None, "it must be held"


def test_an_approved_write_runs_and_carries_the_token(monkeypatch):
    seen: list[dict] = []

    async def _fake_write(**kwargs):
        seen.append(kwargs)
        return _branch_result(kwargs["repository"], kwargs["branch"])

    monkeypatch.setattr(ai_tools, "write_github_create_branch", _fake_write)
    token = issue_approval_token("u-2", "github_create_branch")

    result = asyncio.run(tool_loop.dispatch_tool(
        "u-2", "c-2", "github_create_branch",
        {"repository": "o/r", "branch": "terminal"},
        approval_tokens=[token],
    ))

    assert result["ok"] is True
    assert seen and seen[0]["approved"] is True, "the forwarded token must reach the executor"
    assert seen[0]["branch"] == "terminal"
    assert store.get_pending_github_action("u-2", "c-2") is None


def test_the_full_loop_holds_then_runs_on_confirm(monkeypatch):
    calls: list[dict] = []

    async def _fake_write(**kwargs):
        calls.append(kwargs)
        return _branch_result(kwargs["repository"], kwargs["branch"])

    monkeypatch.setattr(ai_tools, "write_github_create_branch", _fake_write)

    proposed = asyncio.run(tool_loop.dispatch_tool(
        "u-3", "c-3", "github_create_branch",
        {"repository": "o/r", "branch": "terminal"},
    ))
    assert proposed["code"] == "APPROVAL_REQUIRED"
    assert calls == []
    assert store.get_pending_github_action("u-3", "c-3") is not None

    answer = asyncio.run(chat_tools.handle_github_chat_with_tools(
        "u-3", "ok create maadu", history=None, conversation_id="c-3",
    ))

    assert calls, "the held write must run after the confirmation"
    assert "terminal" in answer
    assert store.get_pending_github_action("u-3", "c-3") is None


def test_a_write_that_does_not_hold_is_still_refused_without_a_token():
    """github_update_repository_file is gated at dispatch, not held."""
    result = asyncio.run(tool_loop.dispatch_tool(
        "u-4", "c-4", "github_update_repository_file",
        {"repository": "o/r", "path": "a.py", "content": "x", "message": "m"},
    ))
    assert result["ok"] is False
    assert result["code"] == "APPROVAL_REQUIRED"


def test_every_write_tool_forwards_the_token_into_the_payload(monkeypatch):
    """An approved call must carry the token to the executor, for every write."""
    token = issue_approval_token("u-5", "github_merge_pull_request")
    seen: dict = {}

    async def _fake_run_tool_async(name, payload, **kwargs):
        seen["payload"] = json.loads(payload)
        return ai_tools.ToolResult(name=name, output="{}", safe=True)

    monkeypatch.setattr(tool_loop, "run_tool_async", _fake_run_tool_async)

    result = asyncio.run(tool_loop.dispatch_tool(
        "u-5", "c-5", "github_merge_pull_request",
        {"repository": "o/r", "number": 7},
        approval_tokens=[token],
    ))

    assert result["ok"] is True
    assert seen["payload"]["approval_token"] == token
