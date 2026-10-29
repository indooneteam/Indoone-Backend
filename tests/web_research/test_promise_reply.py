"""A promise is not an answer.

"One moment, I will fetch the latest news" reached the user as the reply, and the fetch
never happened. If a promise still reaches the end of the pipeline, it is replaced by the
work it promised.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.quality.promise import is_promise
from app.web_research import fast_path
from app.web_research.research import ResearchResult
from app.web_research.tools import SearchWebToolResult


@pytest.mark.parametrize(
    "reply",
    [
        "One moment, I will fetch the latest news.",
        "ಒಂದು ಕ್ಷಣ, Jio IPO ಬಗ್ಗೆ ಸಮಾಚಾರ ತರ್ತೀನಿ.",
        "Let me check that for you.",
        "Just a moment.",
        "I'll search for that.",
        "Hold on, fetching it.",
    ],
)
def test_a_promise_is_recognised(reply: str) -> None:
    assert is_promise(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        "The listing date is not confirmed by any official source yet.",
        "Deepavali is on 8 November 2026.",
        "",
        "Here is the answer. " * 30,
    ],
)
def test_an_answer_is_not_a_promise(reply: str) -> None:
    assert is_promise(reply) is False


@pytest.fixture()
def wired(monkeypatch):
    async def fake_search(call):
        return SearchWebToolResult(
            status="ok",
            results=(
                ResearchResult(
                    title="Jio Platforms IPO",
                    url="https://example.com/jio",
                    snippet="The listing date has not been announced.",
                ),
            ),
        )

    async def fake_reply(message, history=None, document_context="", extra_system_instruction="", **kwargs):
        return "No official listing date has been announced."

    monkeypatch.setattr(fast_path, "execute_search_web_call", fake_search)
    monkeypatch.setattr(fast_path, "generate_reply", fake_reply)


def test_a_promise_is_replaced_by_the_answer(wired) -> None:
    replacement = asyncio.run(
        fast_path.answer_instead_of_promise(
            "One moment, I will fetch the latest news.",
            "jio ipo estakke listing agide",
        )
    )

    assert replacement is not None
    assert "No official listing date" in replacement
    assert not is_promise(replacement)


def test_an_answer_is_left_alone(wired) -> None:
    assert asyncio.run(
        fast_path.answer_instead_of_promise("Deepavali is on 8 November.", "when is deepavali")
    ) is None


@pytest.mark.parametrize(
    "message",
    ["unread emails about deepavali", "check the branch for deepavali", "namma maatu helu"],
)
def test_a_promise_about_another_subject_is_left_alone(message: str, wired) -> None:
    """Those have deterministic routers; a web search over one would answer the wrong thing."""
    assert asyncio.run(
        fast_path.answer_instead_of_promise("One moment, let me check.", message)
    ) is None
