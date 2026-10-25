"""Second-pass fixes in the tool layer, each pinned by a test.

These cover: the GitHub file-update payload key, identifier validation on the
GitHub read tools, the sync-tool worker slot being returned on a timeout, the
output cap, and the backend-set approval flag a write executor now reads.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from app.ai.execution import tool_loop
from app.ai.execution import tools as ai_tools
from app.ai.execution.approval import issue_approval_token
from app.ai.execution.tools import TOOLS, ToolResult, run_tool
from app.plugins.github import chat_tools


@pytest.fixture(autouse=True)
def _allow_connectors(monkeypatch):
    # Authorization is exercised by its own tests; here it must not hide the
    # validation and executor behaviour under test.
    allow = lambda *args, **kwargs: SimpleNamespace(allowed=True, reason="")  # noqa: E731
    monkeypatch.setattr("app.ai.execution.tools.authorize_tool", allow)
    monkeypatch.setattr("app.ai.execution.approval.authorize_tool", allow)


@pytest.fixture
def _approval_env(monkeypatch):
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a-test-approval-secret-that-is-long-enough")


def test_chat_tools_sends_the_sha_key_the_executor_reads(monkeypatch):
    captured: dict = {}

    async def fake(name, payload, **kwargs):
        captured["payload"] = json.loads(payload)
        return ToolResult(name=name, output="{}", safe=True)

    monkeypatch.setattr(chat_tools, "run_tool_async", fake)

    asyncio.run(chat_tools._execute_github_tool(
        "u1",
        "github_update_repository_file",
        {
            "repository": "o/r",
            "path": "a.py",
            "content": "x",
            "message": "m",
            "sha": "abc123",
            "approval_token": "tok",
        },
    ))

    assert captured["payload"]["sha"] == "abc123", "the executor reads `sha`, not `expected_sha`"


def test_pull_request_files_requires_a_number():
    result = run_tool("github_list_pull_request_files", json.dumps({"user_id": "u", "repository": "o/r"}), user_id="u")
    assert result.safe is False
    assert "number" in result.output.casefold()


def test_get_commit_requires_a_sha():
    result = run_tool("github_get_commit", json.dumps({"user_id": "u", "repository": "o/r"}), user_id="u")
    assert result.safe is False
    assert "sha" in result.output.casefold()


def test_a_timed_out_sync_tool_returns_its_worker_slot(monkeypatch):
    block = threading.Event()

    def hang(_: str) -> str:
        block.wait()
        return "never"

    try:
        for _ in range(ai_tools.MAX_SYNC_TOOL_WORKERS + 1):
            with pytest.raises(TimeoutError):
                ai_tools._run_sync_with_timeout(hang, "x", 0.02)
    finally:
        block.set()


