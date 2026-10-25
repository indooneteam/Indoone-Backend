"""A GitHub write is proposed, held, and only runs when the user confirms.

The model proposing a write is not the write happening. These tests pin the three
steps - propose, hold, run on confirmation - because that is the rule the whole
GitHub write path rests on.
"""

import asyncio
import json

import pytest
from cryptography.fernet import Fernet

from app.ai.execution.tools import ToolResult

from app.capabilities.connectors import store
from app.plugins.github import chat_tools


@pytest.fixture(autouse=True)
def _clean_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a-test-approval-secret-that-is-long-enough")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    store.initialize()
    yield


def test_a_write_without_a_token_is_held_not_run(monkeypatch):
    """The model calls a write with no approval. Nothing may happen."""
    ran = {"value": False}

    async def _never(name, payload, **kwargs):
        ran["value"] = True
        return {"ok": True, "result": {}}

    monkeypatch.setattr(chat_tools, "run_tool_async", _never)

    result = asyncio.run(chat_tools._execute_github_tool(
        "user-1",
        "github_create_branch",
        {"repository": "o/r", "branch": "terminal", "conversation_id": "c-1"},
    ))

    assert result["ok"] is False
    assert result["code"] == "APPROVAL_REQUIRED"
    assert ran["value"] is False, "the write must not have run"
    # and it is waiting
    pending = store.get_pending_github_action("user-1", "c-1")
    assert pending is not None


def test_confirming_runs_the_write_it_held(monkeypatch):
    calls = []

    async def _record(name, payload, **kwargs):
        calls.append((name, json.loads(payload)))
        return ToolResult(
            name=name,
            safe=True,
            output=json.dumps({"ok": True, "result": {"branch": "terminal", "sha": "a" * 40, "from": "main"}}),
        )

    monkeypatch.setattr(chat_tools, "run_tool_async", _record)

    asyncio.run(chat_tools._execute_github_tool(
        "user-2",
        "github_create_branch",
        {"repository": "o/r", "branch": "terminal", "conversation_id": "c-2"},
    ))

    answer = asyncio.run(chat_tools.handle_github_chat_with_tools(
        "user-2", "ok create maadu", history=None, conversation_id="c-2",
    ))

    assert calls, "the write should have run after the confirmation"
    name, payload = calls[0]
    assert name == "github_create_branch"
    assert payload["branch"] == "terminal"
    assert payload.get("approval_token"), "the held token must travel with the replay"
    assert "terminal" in answer
    # and it is no longer waiting
    assert store.get_pending_github_action("user-2", "c-2") is None


def test_cancelling_drops_it_without_running_anything(monkeypatch):
    ran = {"value": False}

    async def _never(name, payload, **kwargs):
        ran["value"] = True
        return {"ok": True, "result": {}}

    monkeypatch.setattr(chat_tools, "run_tool_async", _never)

    asyncio.run(chat_tools._execute_github_tool(
        "user-3",
        "github_create_branch",
        {"repository": "o/r", "branch": "x", "conversation_id": "c-3"},
    ))
    answer = asyncio.run(chat_tools.handle_github_chat_with_tools(
        "user-3", "beda", history=None, conversation_id="c-3",
    ))

    assert ran["value"] is False
    assert "cancel" in answer.casefold() or "cancelled" in answer.casefold()
    assert store.get_pending_github_action("user-3", "c-3") is None


def test_anything_else_reminds_the_user_and_holds_it(monkeypatch):
    async def _never(name, payload, **kwargs):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(chat_tools, "run_tool_async", _never)

    asyncio.run(chat_tools._execute_github_tool(
        "user-4",
        "github_create_pull_request",
        {"repository": "o/r", "title": "t", "head": "h", "conversation_id": "c-4"},
    ))
    answer = asyncio.run(chat_tools.handle_github_chat_with_tools(
        "user-4", "what is the weather", history=None, conversation_id="c-4",
    ))

    # The prompt names the state, offers a numbered choice so the answer does not
    # depend on knowing a word, and says plainly that nothing has happened yet.
    assert "still waiting" in answer
    assert "Reply 1" in answer and "2" in answer
    assert store.get_pending_github_action("user-4", "c-4") is not None, "it must still be waiting"


def test_a_pending_is_per_conversation(monkeypatch):
    async def _never(name, payload, **kwargs):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(chat_tools, "run_tool_async", _never)

    asyncio.run(chat_tools._execute_github_tool(
        "user-5",
        "github_create_branch",
        {"repository": "o/r", "branch": "x", "conversation_id": "c-5a"},
    ))
    assert store.get_pending_github_action("user-5", "c-5a") is not None
    assert store.get_pending_github_action("user-5", "c-5b") is None


def test_the_write_intents_are_all_known():
    """Every write tool must have wording, or the proposal reads wrong."""
    from app.ai.execution.tool_registry import GITHUB_WRITE_TOOL_DECLARATIONS

    for declaration in GITHUB_WRITE_TOOL_DECLARATIONS:
        assert declaration["name"] in chat_tools._WRITE_INTENT, declaration["name"]
        assert declaration["name"] in chat_tools._WRITE_TOOLS, declaration["name"]
