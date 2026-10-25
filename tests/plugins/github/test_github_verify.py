"""A write is only reported as done when the state was read back and found.

This is the rule that stops the assistant saying "the push went through" when it did not.
The write call returning ok is the provider accepting the request; the read-back is the
evidence. Every check here is a fake read, so the tests say what each read must prove.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.execution.tools import ToolResult
from app.plugins.github import verify


def _reads(monkeypatch, payload: dict | None, *, safe: bool = True):
    """Point the read-back at a fixed answer, and record which tool was asked."""

    asked: list[str] = []

    async def fake_run_tool_async(name, body, **kwargs):
        asked.append(name)
        if payload is None:
            return ToolResult(name=name, output="", safe=safe)
        return ToolResult(name=name, output=json.dumps(payload), safe=True)

    monkeypatch.setattr(verify, "run_tool_async", fake_run_tool_async)
    return asked


def test_a_created_branch_is_confirmed_only_when_the_list_has_it(monkeypatch) -> None:
    _reads(monkeypatch, {"branches": [{"name": "main"}, {"name": "release"}]})

    found = asyncio.run(verify.verify_write(
        "u1", "github_create_branch", {"repository": "o/r", "branch": "release"}, {},
    ))
    missing = asyncio.run(verify.verify_write(
        "u1", "github_create_branch", {"repository": "o/r", "branch": "hotfix"}, {},
    ))

    assert found.confirmed is True and "release" in found.detail
    assert missing.confirmed is False
    assert "does not contain hotfix" in missing.detail


def test_a_deleted_branch_is_confirmed_only_when_it_is_gone(monkeypatch) -> None:
    _reads(monkeypatch, {"branches": [{"name": "main"}]})

    gone = asyncio.run(verify.verify_write(
        "u1", "github_delete_branch", {"repository": "o/r", "branch": "release"}, {},
    ))

    assert gone.confirmed is True
    assert "no longer" in gone.detail


def test_a_delete_is_not_confirmed_while_the_branch_is_still_there(monkeypatch) -> None:
    _reads(monkeypatch, {"branches": [{"name": "release"}]})

    still = asyncio.run(verify.verify_write(
        "u1", "github_delete_branch", {"repository": "o/r", "branch": "release"}, {},
    ))

    assert still.confirmed is False
    assert "still in the branch list" in still.detail


def test_a_pull_request_is_confirmed_from_the_list(monkeypatch) -> None:
    _reads(monkeypatch, {"pull_requests": [{"number": 9, "head": "release", "state": "open"}]})

    found = asyncio.run(verify.verify_write(
        "u1", "github_create_pull_request", {"repository": "o/r", "head": "release"}, {},
    ))

    assert found.confirmed is True


def test_a_merge_is_not_confirmed_while_the_pull_request_is_open(monkeypatch) -> None:
    _reads(monkeypatch, {"pull_requests": [{"number": 9, "state": "open", "merged_at": None}]})

    open_pr = asyncio.run(verify.verify_write(
        "u1", "github_merge_pull_request", {"repository": "o/r", "number": 9}, {},
    ))

    assert open_pr.confirmed is False
    assert "still open" in open_pr.detail


def test_a_merge_is_confirmed_when_the_list_says_merged(monkeypatch) -> None:
    _reads(monkeypatch, {"pull_requests": [{"number": 9, "state": "merged", "merged_at": "2026-10-25"}]})

    merged = asyncio.run(verify.verify_write(
        "u1", "github_merge_pull_request", {"repository": "o/r", "number": 9}, {},
    ))

    assert merged.confirmed is True


def test_a_file_write_is_confirmed_by_reading_the_content_back(monkeypatch) -> None:
    content = "print('hello from indoone')"
    _reads(monkeypatch, {"content": content, "path": "a.py"})

    landed = asyncio.run(verify.verify_write(
        "u1", "github_update_repository_file",
        {"repository": "o/r", "path": "a.py", "content": content}, {},
    ))

    assert landed.confirmed is True


def test_a_file_write_is_not_confirmed_when_the_content_is_old(monkeypatch) -> None:
    _reads(monkeypatch, {"content": "print('old')"})

    stale = asyncio.run(verify.verify_write(
        "u1", "github_update_repository_file",
        {"repository": "o/r", "path": "a.py", "content": "print('new')"}, {},
    ))

    assert stale.confirmed is False


def test_a_read_that_fails_is_reported_as_unconfirmed_not_as_done(monkeypatch) -> None:
    _reads(monkeypatch, None, safe=False)

    result = asyncio.run(verify.verify_write(
        "u1", "github_create_branch", {"repository": "o/r", "branch": "release"}, {},
    ))

    assert result.checked is True
    assert result.confirmed is False
    assert "could not read" in result.detail


def test_the_verification_fields_are_always_present(monkeypatch) -> None:
    """verified is never absent, so a caller cannot read "not checked" as "fine"."""
    from app.ai.execution import tools as ai_tools

    _reads(monkeypatch, {"branches": [{"name": "release"}]})

    async def fake_run_async(awaitable, timeout=None):
        # The real one awaits the write; this one closes it so nothing is left pending.
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        return {"ok": True, "result": {"branch": "release"}}

    monkeypatch.setattr(ai_tools, "_run_async", fake_run_async)

    raw = asyncio.run(ai_tools._github_create_branch(json.dumps({
        "user_id": "u1", "repository": "o/r", "branch": "release", "original_message": "create release",
    })))
    payload = json.loads(raw)

    assert payload["verified"] is True
    assert payload["checked"] is True
    assert payload["verification"]
