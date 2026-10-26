"""The branch list must be the real one, and the count must be the real count.

Two complaints from the product side, and only one of them was a code defect.

*The count.* The list stopped at one page - thirty branches by default - so a
repository with forty was reported as having thirty. A wrong number stated with
confidence is worse than an error, because nothing downstream can tell.

*Creating a branch.* That was already done through the API (POST git/refs), so the
defect was not in the write path at all: it was the instruction, which let the model
answer with git commands instead of calling the tool. The rule is asserted here so it
cannot quietly disappear.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.gateway.ai_rules import TOOL_SYSTEM_INSTRUCTION
from app.plugins.github import service


def _run(coro):
    return asyncio.run(coro)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Client:
    """Answers each page from the list it was given, and records the page numbers asked."""

    def __init__(self, pages):
        self._pages = pages
        self.requested: list[int] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, _url, *, headers=None, params=None):
        page = int((params or {}).get("page", 1))
        self.requested.append(page)
        index = page - 1
        return _Response(self._pages[index] if index < len(self._pages) else [])


def _patch(monkeypatch, pages):
    client = _Client(pages)
    monkeypatch.setattr(service, "_github_token", lambda _user: "token")
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **_kwargs: client)
    return client


def _branch(name: str) -> dict:
    return {"name": name, "commit": {"sha": "a" * 40}, "protected": False}


def test_the_count_is_every_branch_not_the_first_page(monkeypatch) -> None:
    first = [_branch(f"b{index}") for index in range(30)]
    second = [_branch(f"b{index}") for index in range(30, 40)]
    client = _patch(monkeypatch, [first, second])

    result = _run(service.list_github_branches("u1", "owner/repo", per_page=30))

    assert result["count"] == 40
    assert len(result["branches"]) == 40
    assert result["truncated"] is False
    # It asked for the second page, and stopped after the short one.
    assert client.requested == [1, 2]


def test_a_short_page_is_the_last_page(monkeypatch) -> None:
    client = _patch(monkeypatch, [[_branch("main")]])

    result = _run(service.list_github_branches("u1", "owner/repo", per_page=30))

    assert result["count"] == 1
    assert client.requested == [1]


def test_the_cap_is_reported_rather_than_hidden(monkeypatch) -> None:
    # Three full pages against a cap of two pages' worth.
    pages = [[_branch(f"p{page}-{index}") for index in range(10)] for page in range(5)]
    _patch(monkeypatch, pages)
    monkeypatch.setattr(service, "_MAX_BRANCHES", 20)

    result = _run(service.list_github_branches("u1", "owner/repo", per_page=10))

    assert result["count"] == 20
    assert result["truncated"] is True


def test_the_model_is_told_to_use_the_tool_for_branches() -> None:
    instruction = TOOL_SYSTEM_INSTRUCTION

    assert "github_create_branch" in instruction
    assert "Never answer with git commands" in instruction
    assert "never state a branch count" in instruction
