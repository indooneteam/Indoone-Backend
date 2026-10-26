"""The thread is already about a repository; a follow-up should not re-ask which one.

"Branch estu ede adralli" names nothing, and the old answer was "Which repository should I
open? Choose an exact owner/name" — asking the user to repeat what the last two turns had
already settled. The active repository is now bound from recent history.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.execution.reference_resolution import recent_repository
from app.plugins.github import chat_tools

BRANCH_LIST = """Branches in indooneteam/indoone (4 returned):
1. main · a1b2c3d4e5 · protected
2. develop · b2c3d4e5f6
3. terminal · c3d4e5f6a7
4. web · d4e5f6a7b8"""


def test_the_active_repository_is_read_from_the_last_turns() -> None:
    history = [
        ("user", "list my repos"),
        ("assistant", "1. indooneteam/Indoone-Backend (main) — Private"),
        ("user", "show branches"),
        ("assistant", BRANCH_LIST),
    ]

    assert recent_repository(history) == "indooneteam/indoone"


def test_a_repository_named_in_prose_counts_too() -> None:
    history = [
        ("user", "what is in indooneteam/indoone?"),
        ("assistant", "It has four branches."),
    ]

    assert recent_repository(history) == "indooneteam/indoone"


def test_an_old_mention_is_not_reused() -> None:
    history = [
        ("assistant", BRANCH_LIST),
        ("user", "thanks"),
        ("assistant", "You're welcome."),
        ("user", "now something else entirely"),
    ]

    # Outside the three-turn window: the conversation has moved on.
    assert recent_repository(history, turns=2) == ""


def test_no_repository_anywhere_returns_nothing() -> None:
    assert recent_repository([("user", "hello"), ("assistant", "hi")]) == ""
    assert recent_repository(None) == ""


def test_a_follow_up_binds_the_active_repository_instead_of_asking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(user_id, name, arguments, approval_tokens=(), conversation_id=None, message=""):
        captured["name"] = name
        captured["arguments"] = dict(arguments)
        return {"ok": True, "result": {"repository": "indooneteam/indoone", "branches": []}}

    monkeypatch.setattr(chat_tools, "_execute_github_tool", fake_execute)
    monkeypatch.setattr(chat_tools, "_format_github_result", lambda *_a, **_k: "done")

    history = [
        ("user", "list my repos"),
        ("assistant", BRANCH_LIST),
    ]

    reply = asyncio.run(
        chat_tools.handle_github_chat_with_tools(
            "user-one",
            "show branches",
            history=history,
            conversation_id="c1",
        )
    )

    assert reply == "done"
    assert captured["name"] == "github_list_branches"
    # Bound from the thread, not asked for.
    assert captured["arguments"]["repository"] == "indooneteam/indoone"
