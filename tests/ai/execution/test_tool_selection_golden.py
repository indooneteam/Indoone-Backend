"""The golden set that keeps tool discovery honest.

The catalogue is no longer sent whole, so the search is what stands between the model
and every tool it was not handed. That makes two numbers matter, and both are gated here
on every pull request:

* recall - the tool the user needs is in the shortlist. A miss is a capability the model
  cannot reach at all, so this is the quality gate, at 95 percent.
* top-1 - the right tool comes first. When it does, the model almost always takes it.

What is deliberately not gated: a question that needs no tool may still surface a
candidate. "Explain what a pull request is" shares words with five GitHub tools, and a
score threshold cannot tell a definition from a request without becoming the keyword list
this replaced. The cost is bounded - at most five declarations, about 600 tokens - and
the model reads the descriptions and declines. Only a live model run can measure whether
it declined, so that measurement is a separate, opt-in check, not a CI gate.

The cases live in data/eval/tool_selection.jsonl: id, language, the user's message, the
English query the model writes for find_tools, and the tool that must be reachable. The
Indic cases carry the message and the query separately on purpose: the search only ever
sees the query, and the message is there for the live run and for the reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.execution.tool_registry import EMAIL_TOOL_DECLARATIONS, TOOL_SPECS
from app.ai.execution.tool_search import search_tool_names

CASES_PATH = Path(__file__).resolve().parents[3] / "data" / "eval" / "tool_selection.jsonl"
RECALL_GATE = 0.95
TOP_ONE_GATE = 0.90
MINIMUM_CASES = 150


def _cases() -> list[dict[str, object]]:
    assert CASES_PATH.is_file(), f"missing the golden set: {CASES_PATH}"
    cases: list[dict[str, object]] = []
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _catalogue_names() -> set[str]:
    names = {spec.name for spec in TOOL_SPECS}
    names |= {str(item["name"]) for item in EMAIL_TOOL_DECLARATIONS}
    names.add("search_web")
    # find_tools is not a capability a user asks for; it is always in the catalogue, so
    # it needs no case.
    names.discard("find_tools")
    return names


def test_the_set_is_big_enough_and_covers_every_tool() -> None:
    cases = _cases()

    assert len(cases) >= MINIMUM_CASES
    covered = {str(case["expected"]) for case in cases if case["expected"]}
    missing = sorted(_catalogue_names() - covered)
    assert missing == [], f"these tools have no case at all: {missing}"
    assert {str(case["language"]) for case in cases} >= {"en", "kn", "hi", "ta", "mixed"}


def test_every_case_names_a_tool_that_exists() -> None:
    """A case for a tool that was renamed would pass silently, and measure nothing."""
    known = _catalogue_names()
    unknown = sorted({str(case["expected"]) for case in _cases() if case["expected"] and str(case["expected"]) not in known})

    assert unknown == [], f"cases name tools that do not exist: {unknown}"


def test_recall_is_at_least_95_percent() -> None:
    """The tool the user needs must be in the shortlist."""
    wanted = [case for case in _cases() if case["expected"]]
    misses = [
        case
        for case in wanted
        if str(case["expected"]) not in search_tool_names(str(case["query"]), limit=5)
    ]
    recall = 1 - len(misses) / len(wanted)

    assert recall >= RECALL_GATE, (
        f"recall {recall:.1%} is below {RECALL_GATE:.0%}; misses: "
        + "; ".join(f"{case['query']} -> {case['expected']}" for case in misses[:10])
    )


def test_the_right_tool_usually_comes_first() -> None:
    wanted = [case for case in _cases() if case["expected"]]
    first = sum(
        1
        for case in wanted
        if (search_tool_names(str(case["query"]), limit=5) or [""])[0] == str(case["expected"])
    )
    accuracy = first / len(wanted)

    assert accuracy >= TOP_ONE_GATE, f"top-1 accuracy {accuracy:.1%} is below {TOP_ONE_GATE:.0%}"


@pytest.mark.parametrize(
    "greeting",
    ["hi", "hello", "namaskara", "ನಮಸ್ಕಾರ", "नमस्ते", "வணக்கம்", "thanks", "ok"],
)
def test_a_bare_greeting_finds_no_tool(greeting: str) -> None:
    """The one case where zero tools is a promise, not a preference.

    A greeting is the message that must never pay for a catalogue - it was the example
    that started this whole change.
    """
    assert search_tool_names(greeting) == []


def test_a_writing_request_finds_no_tool() -> None:
    for request in ("write a story about a king", "ಒಂದು ಕಥೆ ಬರೆ", "एक कविता लिखो"):
        assert search_tool_names(request) == []


def test_a_definitional_question_is_bounded_not_silent() -> None:
    """A definition may surface candidates; it must not surface many.

    This is the honest boundary of a lexical search: the score cannot separate "what is
    a pull request" from "list my pull requests" without becoming a keyword list. What
    is guaranteed is the cost - at most the five the tool returns.
    """
    names = search_tool_names("explain what a pull request is", limit=5)

    assert len(names) <= 5
