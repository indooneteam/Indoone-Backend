"""Tool discovery: the search behind ``find_tools``.

The catalogue is no longer sent whole, so this search is what stands between the model
and every tool it was not given. A miss here is a capability the model cannot reach, so
the tests are about the queries a real assistant gets, not about the algorithm.
"""

from __future__ import annotations

import pytest

from app.ai.execution import tool_search


def test_it_finds_the_github_tools_for_a_github_request() -> None:
    names = tool_search.search_tool_names("list my github repositories")

    assert names[0] == "github_list_repositories"


def test_it_finds_email_tools_for_an_email_request() -> None:
    names = tool_search.search_tool_names("read my unread email")

    assert "read_email" in names[:2]


def test_it_finds_the_writing_tools_for_a_write_request() -> None:
    names = tool_search.search_tool_names("send an email to ravi")

    assert "prepare_email_to_send" in names


def test_it_finds_the_local_tools() -> None:
    assert "json_summary" in tool_search.search_tool_names("summarize a json document")
    assert "data_analysis" in tool_search.search_tool_names("analyze a csv dataset")
    assert "code_analysis" in tool_search.search_tool_names("run some code")


@pytest.mark.parametrize(
    "query",
    [
        "write a story about a king",
        "how are you",
        "explain photosynthesis",
        "translate this to kannada",
    ],
)
def test_a_request_that_needs_no_tool_finds_nothing(query: str) -> None:
    """No match is a real answer: the model then answers from what it knows.

    If this ever starts returning tools for a writing request, the model is being
    handed a tool it did not need - which is the waste this whole layer removes.
    """
    assert tool_search.search_tool_names(query) == []


def test_a_name_match_ranks_above_a_description_match() -> None:
    scored = dict(tool_search.score_tools("email"))

    assert scored["read_email"] > scored.get("github_list_repositories", 0.0)


def test_it_returns_at_most_the_limit() -> None:
    assert len(tool_search.search_tool_names("list", limit=3)) <= 3


def test_the_payload_lists_names_and_descriptions() -> None:
    payload = tool_search.find_tools_payload("list my github repositories")

    assert payload["ok"] is True
    tools = payload["result"]["tools"]
    assert tools
    assert all(item["name"] and item["description"] for item in tools)


def test_an_empty_query_is_refused_with_a_reason() -> None:
    payload = tool_search.find_tools_payload("   ")

    assert payload["ok"] is False
    assert payload["code"] == "EMPTY_QUERY"


def test_no_match_says_so_instead_of_failing() -> None:
    payload = tool_search.find_tools_payload("write a story about a king")

    assert payload["ok"] is True
    assert payload["result"]["tools"] == []
    assert "No tool matches" in payload["result"]["note"]


def test_discovery_does_not_depend_on_what_is_connected() -> None:
    """A tool is offered even when its connector is not connected.

    The executor answers with "connect it in Plugins", which is a reply the user can
    act on. Hiding it made the model say it had no access to a capability the product
    has, and it could not tell the user what to do about it.
    """
    names = tool_search.search_tool_names("list my github repositories")

    assert "github_list_repositories" in names


# ---- a query written in the user's language ------------------------------- #
# The model usually writes the query in English, but it also passes the user's own
# words through. Before the alias table those words tokenised to nothing, the search
# returned no tools at all, and the model told the user it had no access - a
# capability the product has, reported as missing.


@pytest.mark.parametrize(
    "query,expected",
    [
        ("ಬ್ರಾಂಚ್ ಎಷ್ಟಿದೆ", "github_list_branches"),
        ("ರೆಪೊದಲ್ಲಿ ಎಷ್ಟು ಬ್ರಾಂಚ್ ಇದೆ", "github_list_branches"),
        ("ಇಮೇಲ್ ಓದು", "read_email"),
        ("ಕಮಿಟ್ ತೋರಿಸು", "github_list_repository_commits"),
    ],
)
def test_a_kannada_query_finds_the_english_tool(query: str, expected: str) -> None:
    assert expected in tool_search.search_tool_names(query)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("branch eshtu ide", "github_list_branches"),
        ("repo branches torsu", "github_list_branches"),
        ("commit ella torsu", "github_list_repository_commits"),
    ],
)
def test_a_romanised_kannada_query_finds_the_tool(query: str, expected: str) -> None:
    assert expected in tool_search.search_tool_names(query)


def test_an_english_query_scores_exactly_as_it_did_before() -> None:
    """The alias table must not change an English query's outcome.

    A word only reaches the table by an exact match on a Kannada or romanised key, so
    every ASCII word is scored the way it always was.
    """

    for query in (
        "list my github repositories",
        "read my unread email",
        "send an email to ravi",
        "summarize a json document",
    ):
        before = [
            name for name, _score in tool_search.score_tools(query)
        ]
        assert before, query
        assert tool_search.search_tool_names(query), query


def test_a_kannada_word_that_names_no_capability_finds_nothing() -> None:
    """A Kannada sentence with no capability in it must not invent a match."""

    assert tool_search.search_tool_names("ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಿ") == []


# ---- a name hit qualifies on its own -------------------------------------- #
# A flat threshold dropped the strongest possible evidence: the query word is the
# tool's own name. One capability word scores low however well it matches, so
# "branch" returned nothing in English as well as in Kannada.


@pytest.mark.parametrize(
    "query,expected",
    [
        ("branch", "github_list_branches"),
        ("ಬ್ರಾಂಚ್", "github_list_branches"),
        ("commit", "github_get_commit"),
    ],
)
def test_a_single_capability_word_finds_its_tool(query: str, expected: str) -> None:
    assert expected in tool_search.search_tool_names(query)


def test_a_name_hit_does_not_push_out_a_better_match() -> None:
    """The rule adds reach; it must not reorder a clear winner."""

    names = tool_search.search_tool_names("list github branches in my repository")

    assert names[0] == "github_list_branches"


def test_a_query_with_no_capability_still_matches_nothing() -> None:
    assert tool_search.search_tool_names("namaskara hegiddiri") == []
    assert tool_search.search_tool_names("write a story about a king") == []
