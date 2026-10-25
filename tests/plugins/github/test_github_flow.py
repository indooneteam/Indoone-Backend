"""The GitHub flow: the parser's patterns, the formatter's answers, the approval rule.

The parser is the part of this file that is expensive to replace, because its
patterns were each added for a real message in a real language. These tests pin
what it does today, so a change to it cannot quietly drop a language.
"""

import json

import pytest

from app.plugins.github import chat_tools


# --------------------------------------------------------------------------- #
# The parser: what it routes, and - just as important - what it refuses to route
# --------------------------------------------------------------------------- #

def _action(message, history=None):
    return chat_tools._github_action(message, history)


def test_a_repository_question_routes_to_the_repository_tool():
    assert _action("nanna repos torsi") == ("github_list_repositories", {"page": 1, "per_page": 10})


def test_a_branch_question_routes_to_the_branch_tool():
    result = _action("IndoMail repo alli estu branches ede")
    assert isinstance(result, tuple), result
    assert result[0] == "github_list_branches"
    assert result[1]["repository_hint"] == "IndoMail"


def test_a_checks_question_routes_to_the_checks_tool():
    result = _action("github repo checks tori")
    assert isinstance(result, tuple), result
    assert result[0] == "github_list_check_runs"


def test_a_pull_request_diff_routes_to_the_pr_files_tool():
    result = _action("github repo pull request 12 files tori")
    assert isinstance(result, tuple), result
    assert result[0] == "github_list_pull_request_files"
    assert result[1]["number"] == 12


def test_one_commit_routes_to_the_commit_tool():
    result = _action("github repo commit 093447c tori")
    assert isinstance(result, tuple), result
    assert result[0] == "github_get_commit"
    assert result[1]["sha"] == "093447c"


@pytest.mark.parametrize("message,expected", [
    ("ok alli terminal antha new branch create madi", "github_create_branch"),
    ("create a branch called terminal", "github_create_branch"),
])
def test_a_write_IS_routed_by_the_parser(message, expected):
    """A write must be answered here, not handed to the model.

    It used to return None so the model would handle it. The model then asked for
    permission over and over, because nothing held the request, and it invented
    branch names when it claimed to have created one. The handler answers a write
    deterministically instead: it holds it, asks in fixed words, and the same
    conversation resolves it.
    """
    result = _action(message)
    assert isinstance(result, tuple), result
    assert result[0] == expected, result


@pytest.mark.parametrize("message", [
    "github repo alli hosa branch madi",
    "branch delete madi",
])
def test_a_write_with_no_name_asks_for_one(message):
    """No branch name in the message means a question back, not a guess."""
    result = _action(message)
    assert isinstance(result, str), result


@pytest.mark.parametrize("message,name", [
    ("ok alli terminal antha new branch create madi", "terminal"),
    ("create a branch called terminal", "terminal"),
    ("create a branch called feature-x", "feature-x"),
])
def test_the_branch_name_is_taken_from_the_message(message, name):
    result = _action(message)
    assert isinstance(result, tuple), result
    assert result[1]["branch"] == name, result


@pytest.mark.parametrize("word", [
    "torsi", "torisu", "tori", "helu", "kodu", "madi", "maadi", "madu", "olage", "holage",
])
def test_the_romanised_action_words_are_still_recognised(word):
    """These were each added because a real message failed without them."""
    assert chat_tools._ACTION_RE.search(f"github repo {word}") is not None, word


@pytest.mark.parametrize("word", [
    "ತೋರಿಸು", "ತೋರಿಸಿ", "ಹುಡುಕು", "ನೋಡು", "ಕೊಡು", "ಮಾಡು", "ಮಾಡಿ", "ಶಾಖೆ", "ಬ್ರಾಂಚ್",
])
def test_the_kannada_words_are_still_recognised(word):
    assert chat_tools._ACTION_RE.search(f"github repo {word}") is not None or \
           chat_tools._GITHUB_TARGET_RE.search(word) is not None, word


@pytest.mark.parametrize("word", ["देखो", "देखिए", "खोजें", "पढ़ें", "रिपोजिटरी"])
def test_the_hindi_words_are_still_recognised(word):
    assert chat_tools._ACTION_RE.search(f"github {word}") is not None or \
           chat_tools._GITHUB_TARGET_RE.search(word) is not None, word


@pytest.mark.parametrize("word", ["branch", "branches", "checks", "ci"])
def test_the_new_target_words_are_recognised(word):
    assert chat_tools._GITHUB_TARGET_RE.search(f"github repo {word}") is not None, word


# --------------------------------------------------------------------------- #
# The formatter: every declared tool must come back with data, not a sentence
# --------------------------------------------------------------------------- #

_FORMATTER_SAMPLES = {
    "github_list_repositories": {"repositories": [{"full_name": "o/r", "private": False, "default_branch": "main"}]},
    "github_list_assigned_issues": {"issues": [{"number": 1, "title": "t", "html_url": "u"}]},
    "github_list_pull_requests": {"pull_requests": [{"number": 2, "title": "p"}], "repository": "o/r"},
    "github_list_repository_contents": {"contents": [{"type": "file", "path": "a.py", "size": 3}], "repository": "o/r", "path": "/"},
    "github_read_repository_file": {"content": "x = 1", "repository": "o/r", "path": "a.py"},
    "github_search_repository_code": {"results": [{"path": "a.py"}], "query": "q", "repository": "o/r"},
    "github_list_repository_commits": {"commits": [{"sha": "a" * 40, "message": "m"}], "repository": "o/r"},
    "github_compare_repository_commits": {"repository": "o/r", "base": "main", "head": "dev", "files": [{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 2}]},
    "github_list_branches": {"branches": [{"name": "main", "sha": "b" * 40, "protected": True}], "repository": "o/r"},
    "github_list_pull_request_files": {"files": [{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 2}], "repository": "o/r", "number": 4},
    "github_list_check_runs": {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}], "repository": "o/r", "ref": "main", "conclusion": "success"},
    "github_get_commit": {"sha": "c" * 40, "repository": "o/r", "message": "m", "author": "a", "date": "d", "files": [{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0}]},
    "github_create_branch": {"repository": "o/r", "result": {"branch": "terminal", "sha": "d" * 40, "from": "main"}},
    "github_delete_branch": {"repository": "o/r", "result": {"branch": "terminal", "deleted": True}},
    "github_create_pull_request": {"repository": "o/r", "result": {"number": 9, "html_url": "u", "title": "t", "head": "terminal", "base": "main"}},
    "github_merge_pull_request": {"repository": "o/r", "result": {"number": 9, "merged": True, "sha": "e" * 40, "method": "squash"}},
}


@pytest.mark.parametrize("tool,data", sorted(_FORMATTER_SAMPLES.items()))
def test_every_tool_gets_a_formatted_answer(tool, data):
    """The old formatter ended with "GitHub request completed." and dropped the
    data, which is why a branch list came back empty. Every tool must now return
    text that carries its own data."""
    answer = chat_tools._format_github_result(tool, {"ok": True, "result": data})
    assert answer and answer.strip(), tool
    assert answer != "GitHub request completed.", tool


def test_a_branch_list_carries_the_branch_names():
    answer = chat_tools._format_github_result(
        "github_list_branches",
        {"ok": True, "result": {"repository": "o/r", "branches": [{"name": "terminal", "sha": "f" * 40}]}},
    )
    assert "terminal" in answer


def test_an_unknown_tool_hands_over_the_data_not_a_sentence():
    answer = chat_tools._format_github_result(
        "github_something_new", {"ok": True, "result": {"key": "value"}}
    )
    assert "key" in answer


def test_a_failed_result_reports_its_message():
    answer = chat_tools._format_github_result("github_list_branches", {"ok": False, "message": "not connected"})
    assert "not connected" in answer


# --------------------------------------------------------------------------- #
# The approval rule
# --------------------------------------------------------------------------- #

def test_the_write_tools_are_declared_and_offered():
    from app.ai.execution.tool_registry import GITHUB_TOOL_DECLARATIONS, GITHUB_WRITE_TOOL_DECLARATIONS

    read_names = {d["name"] for d in GITHUB_TOOL_DECLARATIONS}
    write_names = {d["name"] for d in GITHUB_WRITE_TOOL_DECLARATIONS}
    assert "github_list_branches" in read_names
    assert {"github_create_branch", "github_delete_branch",
            "github_create_pull_request", "github_merge_pull_request"} <= write_names
    # and a read-only query must never pick a write up by accident
    assert not (read_names & write_names)


def test_the_write_tools_are_marked_as_needing_approval():
    from app.ai.execution.tool_registry import TOOL_SPECS

    for spec in TOOL_SPECS:
        if spec.name in {"github_create_branch", "github_delete_branch",
                         "github_create_pull_request", "github_merge_pull_request"}:
            assert spec.requires_approval is True, spec.name
