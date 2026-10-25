"""A write to a protected branch is confirmed, even when it can be undone.

Reversibility was the whole rule before this: a write that changes nothing that cannot
be changed again ran without a confirmation. That is right for a scratch branch and
wrong for `main`, because the blast radius is different - `main` is what deploys and
what everyone branches from, and a mistake there is noticed last.

These tests hold the second gate: the target branch, not only the tool.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.execution import tools as ai_tools
from app.plugins.github import branch_protection


# ---- the branch question --------------------------------------------------- #


def _run(coro):
    return asyncio.run(coro)


def test_a_plain_branch_needs_no_confirmation(monkeypatch) -> None:
    async def not_protected(*_args, **_kwargs):
        return False

    monkeypatch.setattr(branch_protection, "is_protected", not_protected)

    assert _run(branch_protection.protected_branch_reason("u1", "o/r", "feature/x")) == ""


def test_a_protected_branch_says_so_in_words_a_person_can_act_on(monkeypatch) -> None:
    async def protected(*_args, **_kwargs):
        return True

    monkeypatch.setattr(branch_protection, "is_protected", protected)

    reason = _run(branch_protection.protected_branch_reason("u1", "o/r", "main"))

    assert "main" in reason
    assert "protected" in reason
    # Nothing has happened yet, and the sentence says so.
    assert "nothing has been changed" in reason


def test_an_empty_branch_is_resolved_to_the_default_branch(monkeypatch) -> None:
    seen: list[str] = []

    async def default_branch(*_args, **_kwargs):
        return "main"

    async def protected(_user, _repo, branch):
        seen.append(branch)
        return True

    monkeypatch.setattr(branch_protection, "default_branch", default_branch)
    monkeypatch.setattr(branch_protection, "is_protected", protected)

    reason = _run(branch_protection.protected_branch_reason("u1", "o/r", ""))

    assert seen == ["main"]
    assert "main" in reason


def test_an_unknown_target_branch_is_not_allowed_to_run_unconfirmed(monkeypatch) -> None:
    """No branch named and the default cannot be read: we do not know what it touches."""

    async def no_default(*_args, **_kwargs):
        return ""

    monkeypatch.setattr(branch_protection, "default_branch", no_default)

    reason = _run(branch_protection.protected_branch_reason("u1", "o/r", ""))

    assert reason
    assert "confirmation" in reason


def test_the_check_fails_closed_when_github_cannot_answer(monkeypatch) -> None:
    """An expired token must not turn into an unconfirmed write to main."""

    class _Boom:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr(branch_protection.httpx, "AsyncClient", _Boom)

    assert _run(branch_protection.is_protected("u1", "o/r", "main")) is True


def test_a_missing_branch_is_not_protected(monkeypatch) -> None:
    """A branch that does not exist yet is something being created, not damaged."""

    class _Response:
        status_code = 404

        def raise_for_status(self) -> None:  # pragma: no cover - not reached on 404
            raise AssertionError("404 must not raise")

        def json(self):
            return {}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(branch_protection.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(branch_protection, "_github_token", lambda _user: "token")

    assert _run(branch_protection.is_protected("u1", "o/r", "new-branch")) is False


# ---- the gate itself ------------------------------------------------------- #


def test_explicit_intent_does_not_approve_a_protected_write(monkeypatch) -> None:
    """The user asking for it is not a confirmation when the branch is protected."""

    monkeypatch.setattr(ai_tools, "validate_approval_token", lambda *_a, **_k: False)

    data = {"repository": "o/r", "branch": "main", "original_message": "commit this to main"}

    assert ai_tools._write_approved(
        data, "u1", "github_update_repository_file", protected_reason="protected branch"
    ) is False


def test_the_same_write_without_a_protected_reason_still_runs(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "validate_approval_token", lambda *_a, **_k: False)

    data = {"repository": "o/r", "branch": "feature/x", "original_message": "commit this"}

    assert ai_tools._write_approved(data, "u1", "github_update_repository_file") is True


def test_a_token_still_approves_a_protected_write(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "validate_approval_token", lambda *_a, **_k: True)

    data = {"repository": "o/r", "branch": "main", "approval_token": "t"}

    assert ai_tools._write_approved(
        data, "u1", "github_update_repository_file", protected_reason="protected branch"
    ) is True


def test_a_held_protected_write_says_why_it_is_asking(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "validate_approval_token", lambda *_a, **_k: False)
    # tools.py binds the issuer at import, so patching the approval module would not
    # reach the name the executor actually calls.
    monkeypatch.setattr(ai_tools, "issue_approval_token", lambda *_a, **_k: "issued-token")
    monkeypatch.setattr(
        "app.capabilities.connectors.store.save_pending_github_action", lambda *_a, **_k: None
    )

    class _Fernet:
        def encrypt(self, value: bytes) -> bytes:
            return value

    monkeypatch.setattr("app.capabilities.connectors.integrations._fernet", lambda: _Fernet())

    held = ai_tools._hold_write(
        {"repository": "o/r", "branch": "main", "original_message": "commit this to main"},
        "u1",
        "github_update_repository_file",
        protected_reason="This writes directly to main, which is a protected branch.",
    )

    assert held is not None
    payload = json.loads(held)
    assert payload["code"] == "APPROVAL_REQUIRED"
    assert "protected branch" in payload["reply"]


@pytest.mark.parametrize("tool", ["github_create_branch", "github_create_pull_request"])
def test_creating_something_is_not_a_protected_branch_write(tool: str) -> None:
    """A branch creation and a pull request do not modify an existing branch."""

    from app.plugins.github import chat_tools

    assert tool not in chat_tools._PROTECTED_BRANCH_WRITES


def test_the_writes_that_can_land_on_a_protected_branch_are_listed() -> None:
    from app.plugins.github import chat_tools

    assert chat_tools._PROTECTED_BRANCH_WRITES == frozenset(
        {
            "github_update_repository_file",
            "github_delete_branch",
            "github_merge_pull_request",
        }
    )
