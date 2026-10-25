"""An explicit request from the user is the approval.

The assistant used to ask the user to confirm the thing they had just asked for,
and a held proposal could even be reported as done. Now the user's own wording is
read: an explicit "create a branch called test" runs immediately, anything else
still holds, and a held write is never reported as a change that happened.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from app.ai.execution import tool_loop
from app.ai.execution import tools as ai_tools
from app.ai.execution.tools import _explicit_write_intent
from app.capabilities.connectors import store
from app.plugins.github import chat_tools


def _branch_result(repository: str, branch: str) -> dict:
    return {"repository": repository, "result": {"branch": branch, "sha": "a" * 40, "from": "main"}}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a-test-approval-secret-that-is-long-enough")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    allow = lambda *args, **kwargs: SimpleNamespace(allowed=True, reason="")  # noqa: E731
    monkeypatch.setattr("app.ai.execution.connector_authorization.authorize_tool", allow)
    monkeypatch.setattr("app.ai.execution.tools.authorize_tool", allow)
    monkeypatch.setattr("app.ai.execution.approval.authorize_tool", allow)
    store.initialize()
    yield


def test_explicit_intent_is_recognised_per_write():
    assert _explicit_write_intent("github_create_branch", "create a branch called test")
    assert _explicit_write_intent("github_create_branch", "terminal antha branch create madi")
    assert _explicit_write_intent("github_delete_branch", "delete the branch terminal")
    assert _explicit_write_intent("github_create_pull_request", "open a pull request")
    assert _explicit_write_intent("github_merge_pull_request", "merge the pull request")


def test_a_question_is_not_an_explicit_request():
    assert not _explicit_write_intent("github_create_branch", "how many branches are there")
    assert not _explicit_write_intent("github_create_branch", "")
    assert not _explicit_write_intent("github_create_branch", "show me the branches")


def test_an_explicit_request_runs_without_a_confirmation(monkeypatch):
    seen: list[dict] = []

    async def _fake_write(**kwargs):
        seen.append(kwargs)
        return _branch_result(kwargs["repository"], kwargs["branch"])

    monkeypatch.setattr(ai_tools, "write_github_create_branch", _fake_write)

    result = asyncio.run(tool_loop.dispatch_tool(
        "u-1", "c-1", "github_create_branch",
        {"repository": "o/r", "branch": "test"},
        original_message="create a branch called test in o/r",
    ))

    assert result["ok"] is True
    assert seen and seen[0]["approved"] is True
    assert store.get_pending_github_action("u-1", "c-1") is None, "no confirmation is held"


def test_a_proposal_without_an_explicit_request_is_still_held(monkeypatch):
    async def _must_not_run(**kwargs):
        raise AssertionError("the write must not run")

    monkeypatch.setattr(ai_tools, "write_github_create_branch", _must_not_run)

    result = asyncio.run(tool_loop.dispatch_tool(
        "u-2", "c-2", "github_create_branch",
        {"repository": "o/r", "branch": "test"},
        original_message="what branches do I have",
    ))

    assert result["ok"] is False
    assert result["code"] == "APPROVAL_REQUIRED"
    assert store.get_pending_github_action("u-2", "c-2") is not None


def test_the_handler_runs_an_explicit_request_without_holding(monkeypatch):
    calls: list[tuple] = []

    async def _record(name, payload, **kwargs):
        calls.append((name, json.loads(payload)))
        return ai_tools.ToolResult(name=name, safe=True, output=json.dumps(_branch_result("o/r", "test")))

    monkeypatch.setattr(chat_tools, "run_tool_async", _record)

    result = asyncio.run(chat_tools._execute_github_tool(
        "u-3", "github_create_branch",
        {"repository": "o/r", "branch": "test"},
        message="create a branch called test in o/r",
    ))

    assert calls, "an explicit request must run"
    assert result["ok"] is True


def test_a_held_write_is_never_reported_as_done():
    held = {
        "ok": False,
        "code": "APPROVAL_REQUIRED",
        "reply": "I will create a new branch in o/r. Nothing has been changed yet.",
    }
    text = chat_tools._format_github_result("github_create_branch", {"ok": True, "result": held})
    assert "Nothing has been changed yet" in text
    assert "was created" not in text
