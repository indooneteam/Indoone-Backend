"""A branch request in ordinary Kannada reaches the tool, not the model.

"1 ralli" names no verb and no repository. It used to fall through to the model, which
answered with branch names that never existed - the single most damaging failure in this
plugin, because a fabricated branch reads exactly like a real one.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.gateway.ai_rules import TOOL_SYSTEM_INSTRUCTION
from app.plugins.github import chat_tools

REPO_LIST = """Most recently updated accessible GitHub repositories:
1. indoone
https://github.com/indooneteam/indoone

2. Indoone-Backend
https://github.com/indooneteam/Indoone-Backend"""

HISTORY = [("user", "list my repos"), ("assistant", REPO_LIST)]


@pytest.mark.parametrize(
    "message",
    ["1 ralli", "1 alli branch", "branches in 1", "2 ralli branches", "show branches"],
)
def test_a_branch_request_routes_to_the_tool(message: str) -> None:
    action = chat_tools._github_action(message, HISTORY)

    assert action is not None, "the request must not reach the model"
    assert action[0] == "github_list_branches"


def test_a_message_that_is_not_about_github_still_reaches_the_model() -> None:
    assert chat_tools._github_action("hello there", HISTORY) is None


def test_the_bare_ordinal_binds_the_repository_it_points_at(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_execute(user_id, name, arguments, approval_tokens=(), conversation_id=None, message=""):
        captured["name"] = name
        captured["arguments"] = dict(arguments)
        return {"ok": True, "result": {"repository": "indooneteam/indoone", "branches": []}}

    monkeypatch.setattr(chat_tools, "_execute_github_tool", fake_execute)
    monkeypatch.setattr(chat_tools, "_format_github_result", lambda *_a, **_k: "1. main\n2. develop")

    reply = asyncio.run(
        chat_tools.handle_github_chat_with_tools(
            "user-one", "1 ralli", history=HISTORY, conversation_id="c1"
        )
    )

    assert reply == "1. main\n2. develop"
    assert captured["name"] == "github_list_branches"
    assert captured["arguments"]["repository"] == "indooneteam/indoone"


def test_the_instruction_forbids_invented_names() -> None:
    assert "Never invent a branch" in TOOL_SYSTEM_INSTRUCTION
