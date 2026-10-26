"""Typing "2" after a numbered list works.

A branch list came back numbered and the next request said "create a branch in 2". The
router demanded `owner/name`, refused, and the user had to retype a name they had just
been shown. This is the resolver that reads the previous answer and makes the selection
mean what the user meant.

It resolves a value that IS a selection. It never guesses a verb, and it never picks
between two matches - that would be a guess dressed up as an answer.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.execution.reference_resolution import (
    parse_numbered_items,
    resolve_from_history,
    resolve_selection,
)
from app.plugins.github import chat_tools

BRANCH_LIST = """Branches in indooneteam/indoone (3 returned):
1. main · a1b2c3d4e5 · protected
2. develop · b2c3d4e5f6
3. terminal · c3d4e5f6a7"""

REPO_LIST = """Most recently updated accessible GitHub repositories (3 returned):
1. indooneteam/indoone (main) — Public
2. indooneteam/Indoone-Backend (develop) — Private
3. other-org/site (main) — Public"""


def test_items_are_read_without_their_annotations() -> None:
    assert parse_numbered_items(BRANCH_LIST) == ["main", "develop", "terminal"]
    assert parse_numbered_items(REPO_LIST) == [
        "indooneteam/indoone",
        "indooneteam/Indoone-Backend",
        "other-org/site",
    ]


def test_a_number_picks_that_item() -> None:
    items = parse_numbered_items(BRANCH_LIST)

    assert resolve_selection("2", items) == "develop"
    assert resolve_selection("#2", items) == "develop"
    assert resolve_selection("2.", items) == "develop"


def test_a_word_ordinal_picks_the_same_item() -> None:
    items = parse_numbered_items(BRANCH_LIST)

    assert resolve_selection("second", items) == "develop"
    assert resolve_selection("the first one", items) == "main"
    assert resolve_selection("2nd", items) == "develop"
    assert resolve_selection("last", items) == "terminal"


def test_a_partial_name_resolves_only_when_it_is_unambiguous() -> None:
    items = parse_numbered_items(REPO_LIST)

    assert resolve_selection("site", items) == "other-org/site"
    assert resolve_selection("backend", items) == "indooneteam/Indoone-Backend"
    # Two items contain "indoone"; picking one of them would be a guess.
    assert resolve_selection("indoone", items) is None


def test_a_number_past_the_end_resolves_to_nothing() -> None:
    items = parse_numbered_items(BRANCH_LIST)

    assert resolve_selection("9", items) is None
    assert resolve_selection("", items) is None


def test_history_is_read_from_the_assistant_not_the_user() -> None:
    history = [
        ("user", "branch list kodu"),
        ("assistant", BRANCH_LIST),
        ("user", "2"),
    ]

    assert resolve_from_history("2", history) == "develop"
    # The user's own words are what is being resolved; reading them back would resolve a
    # value to itself.
    assert resolve_from_history("branch list kodu", history) is None


def test_the_router_resolves_a_selection_before_executing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(user_id, name, arguments, approval_tokens=(), conversation_id=None, message=""):
        captured["name"] = name
        captured["arguments"] = dict(arguments)
        return {"ok": True, "result": {"branch": "develop", "sha": "b" * 40}}

    monkeypatch.setattr(chat_tools, "_execute_github_tool", fake_execute)
    monkeypatch.setattr(chat_tools, "_format_github_result", lambda *_a, **_k: "done")

    history = [("user", "list repos"), ("assistant", REPO_LIST)]

    reply = asyncio.run(
        chat_tools.handle_github_chat_with_tools(
            "user-one",
            "show branches of 2",
            history=history,
            conversation_id="c1",
        )
    )

    assert reply == "done"
    assert captured["name"] == "github_list_branches"
    # "2" became the second repository, not the literal string "2".
    assert captured["arguments"]["repository"] == "indooneteam/Indoone-Backend"


def test_an_unresolvable_selection_is_left_alone(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(user_id, name, arguments, approval_tokens=(), conversation_id=None, message=""):
        captured["arguments"] = dict(arguments)
        return {"ok": True, "result": {"branch": "x", "sha": "b" * 40}}

    monkeypatch.setattr(chat_tools, "_execute_github_tool", fake_execute)
    monkeypatch.setattr(chat_tools, "_format_github_result", lambda *_a, **_k: "done")

    asyncio.run(
        chat_tools.handle_github_chat_with_tools(
            "user-one",
            "show branches of 9",
            history=[("user", "list repos"), ("assistant", REPO_LIST)],
            conversation_id="c1",
        )
    )

    # Past the end of the list: nothing is invented. The value is passed through and the
    # ordinary validation path answers with its own message.
    assert not str(captured["arguments"].get("repository") or "").strip()
