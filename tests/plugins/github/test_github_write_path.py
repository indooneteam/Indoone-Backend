"""The real write path: the executor holds, the confirmation runs it.

The earlier tests called the GitHub handler directly. A write does not always go
through the handler - the unified loop's executor is where it actually lands - so
these tests drive that executor, which is the path the app uses.
"""

import asyncio
import json

import pytest
from cryptography.fernet import Fernet

from app.ai.execution import tools as ai_tools
from app.ai.execution.tools import ToolResult
from app.capabilities.connectors import store
from app.plugins.github import chat_tools


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a-test-approval-secret-that-is-long-enough")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    store.initialize()
    yield


def test_the_executor_holds_a_write_with_no_token(monkeypatch):
    """This is the path the app uses. It must hold, not refuse."""
    ran = {"value": False}

    async def _never(*args, **kwargs):
        ran["value"] = True
        raise AssertionError("the write must not run before confirmation")

    monkeypatch.setattr(ai_tools, "_run_async", _never)

    payload = json.dumps({
        "user_id": "u-1",
        "conversation_id": "c-1",
        "repository": "teaminmail/IndoMail",
        "branch": "terminal",
    })
    answer = asyncio.run(ai_tools._github_create_branch(payload))

    assert ran["value"] is False
    parsed = json.loads(answer)
    assert parsed["code"] == "APPROVAL_REQUIRED", parsed
    assert store.get_pending_github_action("u-1", "c-1") is not None, "it must be held"


def test_a_held_write_runs_when_the_user_confirms(monkeypatch):
    calls = []

    async def _record(name, payload, **kwargs):
        calls.append((name, json.loads(payload)))
        return ToolResult(
            name=name,
            output=json.dumps({"ok": True, "result": {"branch": "terminal", "sha": "a" * 40, "from": "main"}}),
        )

    # the handler calls run_tool_async, so mock it there - the same level the
    # other github tests use - and the authorization layer is not reached.
    monkeypatch.setattr(chat_tools, "run_tool_async", _record)
    monkeypatch.setattr(ai_tools, "_run_async", _record)

    asyncio.run(ai_tools._github_create_branch(json.dumps({
        "user_id": "u-2",
        "conversation_id": "c-2",
        "repository": "teaminmail/IndoMail",
        "branch": "terminal",
    })))
    assert store.get_pending_github_action("u-2", "c-2") is not None

    # the confirmation, through the handler that chat.py calls
    async def _execute_through_handler():
        return await chat_tools.handle_github_chat_with_tools(
            "u-2", "ok create maadu", history=None, conversation_id="c-2",
        )

    answer = asyncio.run(_execute_through_handler())

    assert calls, "the held write must run after the confirmation"
    assert "terminal" in answer
    assert store.get_pending_github_action("u-2", "c-2") is None, "it must be cleared"


def test_the_held_arguments_carry_the_token_not_the_user_id(monkeypatch):
    async def _never(*args, **kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(ai_tools, "_run_async", _never)
    asyncio.run(ai_tools._github_create_branch(json.dumps({
        "user_id": "u-3", "conversation_id": "c-3", "repository": "o/r", "branch": "x",
    })))

    pending = store.get_pending_github_action("u-3", "c-3")
    assert pending is not None
    from app.capabilities.connectors.integrations import _fernet
    stored = json.loads(_fernet().decrypt(bytes(pending["encrypted_payload"])).decode("utf-8"))
    assert stored["tool"] == "github_create_branch"
    assert stored["token"], "the token must be held with it"
    assert "user_id" not in stored["arguments"], "the user id is not part of the stored action"


def test_a_missing_secret_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities2.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("INDOONE_APPROVAL_SECRET", raising=False)
    store.initialize()

    async def _never(*args, **kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(ai_tools, "_run_async", _never)
    answer = asyncio.run(ai_tools._github_create_branch(json.dumps({
        "user_id": "u-4", "conversation_id": "c-4", "repository": "o/r", "branch": "x",
    })))
    assert json.loads(answer)["code"] == "APPROVAL_UNAVAILABLE"
