"""DuckDuckGo search: parsed cleanly, and not declared on every turn.

The search is free and keyless, so the only things that can go wrong are the parsing and
the cost of offering it. Both are held here: a mocked results page, and the fact that
`search_web` is no longer in the always-on set.
"""

from __future__ import annotations

import pytest

from app.ai.execution.tool_loop import ALWAYS_ON_TOOL_NAMES
from app.web_research.research import (
    DuckDuckGoResearchProvider,
    parse_duckduckgo_results,
)

# The shape DuckDuckGo returns: every link is wrapped in a redirect whose real target is
# the `uddg` parameter, and the text carries markup and entities.
RESULTS_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha&amp;rut=abc">
     Alpha <b>result</b>
  </a>
  <a class="result__snippet" href="#">Alpha is the first thing &amp; it matters.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fbeta&amp;rut=def">
     Beta result
  </a>
  <a class="result__snippet" href="#">Beta is the second thing.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.net/gamma">Gamma result</a>
</div>
"""


def test_results_are_parsed_with_titles_urls_and_snippets() -> None:
    results = parse_duckduckgo_results(RESULTS_HTML, limit=5)

    assert [item.title for item in results] == ["Alpha result", "Beta result", "Gamma result"]
    # The redirect is unwrapped to the real URL, and the entity is decoded.
    assert results[0].url == "https://example.com/alpha"
    assert results[1].url == "https://example.org/beta"
    # A direct link is kept as it is.
    assert results[2].url == "https://example.net/gamma"
    assert results[0].snippet == "Alpha is the first thing & it matters."
    # A result with no snippet is still a result.
    assert results[2].snippet == ""


def test_the_limit_is_respected() -> None:
    assert len(parse_duckduckgo_results(RESULTS_HTML, limit=2)) == 2


def test_duplicates_and_empty_markup_are_handled() -> None:
    doubled = RESULTS_HTML + RESULTS_HTML

    assert len(parse_duckduckgo_results(doubled, limit=10)) == 3
    assert parse_duckduckgo_results("", limit=5) == []
    assert parse_duckduckgo_results("<html><body>nothing here</body></html>") == []


def test_a_short_timeout_is_enforced_at_construction() -> None:
    with pytest.raises(ValueError):
        DuckDuckGoResearchProvider(timeout=0)

    with pytest.raises(ValueError):
        DuckDuckGoResearchProvider(timeout=120)

    assert DuckDuckGoResearchProvider().timeout == 6.0


def test_search_web_is_not_declared_on_every_turn() -> None:
    # The declaration cost a hundred-odd tokens per turn for a tool most messages never
    # need. It stays reachable through find_tools.
    assert "search_web" not in ALWAYS_ON_TOOL_NAMES
    assert "find_tools" in ALWAYS_ON_TOOL_NAMES
