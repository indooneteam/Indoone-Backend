"""A request for web research must reach the search tool.

Reported: asking for web research returned a different answer. The tool was working - a
live query returned current results - but search_web was no longer always-on, and the model
had to call find_tools before it could see the tool. It often did not, so an explicit
request for research was answered from training memory.

The gate removes that dependency: when the message asks for research or names the web, the
declaration is there from the first round.
"""

from __future__ import annotations

import pytest

from app.ai.execution.tool_loop import model_tool_declarations
from app.web_research.tools import wants_web_research


def _names(declarations: list[dict]) -> list[str]:
    return [str(item.get("name")) for item in declarations]


@pytest.mark.parametrize(
    "message",
    [
        "web research maadu",
        "web research",
        "search the web for Karnataka Rajyotsava",
        "internet alli huduku",
        "ಜಾಲದಲ್ಲಿ ಹುಡುಕು",
        "latest news about bengaluru",
        "research madi",
        "google it",
    ],
)
def test_a_research_request_is_recognised(message: str) -> None:
    assert wants_web_research(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "",
        "search my emails",
        "search the repository",
        "unread emails",
        "1 ralli",
        "check the text",
        "what did we discuss",
    ],
)
def test_another_place_is_not_the_web(message: str) -> None:
    assert wants_web_research(message) is False


def test_the_declaration_follows_the_gate() -> None:
    assert "search_web" not in _names(model_tool_declarations(""))
    assert "search_web" in _names(model_tool_declarations("", wants_web=True))


def test_the_always_on_set_is_still_small() -> None:
    """The reason search_web left the always-on set: 41 declarations cost ~4,863 tokens."""
    plain = _names(model_tool_declarations(""))

    assert "find_tools" in plain
    assert "search_web" not in plain
    assert len(plain) < 10
