"""A research request is answered in the turn it is asked.

From the app: "web research maadu jio ipo estakke listing agide" was answered with
"One moment, I will fetch the latest news" - and nothing was fetched. The user had to send
"kodu" to get the answer.

The search now runs before the model speaks, so the promise has nothing to stand on. These
tests hold that shape: one reply, grounded, never a promise - including when the search
fails, which is exactly when a model is most tempted to promise.
"""

from __future__ import annotations

import asyncio

import pytest

from app.web_research import fast_path
from app.web_research.research import ResearchResult
from app.web_research.tools import SearchWebToolResult

PROMISES = ("one moment", "ಒಂದು ಕ್ಷಣ", "i will fetch", "i'll fetch", "kodu", "fetch it")


def _result(*, status: str = "ok", count: int = 2) -> SearchWebToolResult:
    items = tuple(
        ResearchResult(
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            snippet=f"Snippet {index} about the listing date.",
        )
        for index in range(count)
    )
    return SearchWebToolResult(status=status, results=items)


@pytest.fixture()
def wired(monkeypatch):
    """Stub the network and the model, and record what the model was given."""
    seen: dict[str, str] = {}

    async def fake_search(call):
        seen["query"] = call.query
        return _result()

    async def fake_reply(message, history=None, document_context="", extra_system_instruction="", **kwargs):
        seen["context"] = document_context
        seen["instruction"] = extra_system_instruction
        return "The listing date is not confirmed by any official source yet."

    monkeypatch.setattr(fast_path, "execute_search_web_call", fake_search)
    monkeypatch.setattr(fast_path, "generate_reply", fake_reply)
    return seen


def test_a_research_request_is_answered_in_one_turn(wired) -> None:
    answer = asyncio.run(
        fast_path.web_research_answer("web research maadu jio ipo estakke listing agide")
    )

    assert answer is not None
    assert "listing date" in answer
    assert "Sources:" in answer
    assert "https://example.com/0" in answer


def test_the_search_runs_before_the_model_speaks(wired) -> None:
    asyncio.run(fast_path.web_research_answer("web research maadu jio ipo estakke listing agide"))

    # the query the search received, not the whole sentence
    assert "jio ipo" in wired["query"].casefold()
    assert "maadu" not in wired["query"].casefold()
    # and the model was handed the results, not asked to go and get them
    assert "<research>" in wired["context"]
    assert "ALREADY" in wired["instruction"]


@pytest.mark.parametrize(
    "message",
    [
        "web research maadu jio ipo estakke listing agide",
        "search the web for Karnataka Rajyotsava 2026 date",
        "latest news about bengaluru",
    ],
)
def test_no_reply_is_a_promise(message: str, wired) -> None:
    answer = asyncio.run(fast_path.web_research_answer(message)) or ""

    assert not any(promise in answer.casefold() for promise in PROMISES)


def test_an_empty_search_is_said_plainly(monkeypatch) -> None:
    async def fake_search(call):
        return _result(count=0)

    monkeypatch.setattr(fast_path, "execute_search_web_call", fake_search)

    answer = asyncio.run(fast_path.web_research_answer("web research maadu something obscure")) or ""

    assert "no source came back" in answer
    assert not any(promise in answer.casefold() for promise in PROMISES)


def test_a_provider_failure_is_not_a_promise(monkeypatch) -> None:
    async def exploding(call):
        raise RuntimeError("provider down")

    monkeypatch.setattr(fast_path, "execute_search_web_call", exploding)

    answer = asyncio.run(fast_path.web_research_answer("web research maadu gold rate")) or ""

    assert "failed" in answer
    assert not any(promise in answer.casefold() for promise in PROMISES)


@pytest.mark.parametrize(
    "message",
    ["hi", "unread emails", "1 ralli", "what did we discuss", "check the text"],
)
def test_an_ordinary_message_is_left_alone(message: str) -> None:
    assert asyncio.run(fast_path.web_research_answer(message)) is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("web research maadu jio ipo estakke listing agide", "jio ipo estakke listing agide"),
        ("search the web for Karnataka Rajyotsava 2026 date", "Karnataka Rajyotsava 2026 date"),
        ("internet alli huduku today gold rate bengaluru", "today gold rate bengaluru"),
        ("ಜಾಲದಲ್ಲಿ ಹುಡುಕು ರಾಜ್ಯೋತ್ಸವ ದಿನಾಂಕ", "ರಾಜ್ಯೋತ್ಸವ ದಿನಾಂಕ"),
    ],
)
def test_the_command_words_are_not_the_query(message: str, expected: str) -> None:
    assert fast_path.search_query_from(message) == expected
