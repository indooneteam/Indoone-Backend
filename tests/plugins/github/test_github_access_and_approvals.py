"""The three reported defects, held in place.

1. A read never asks for permission - that was already true, and a contract test
   elsewhere enforces it. What was missing is the other half of the complaint: a write
   the user had already confirmed asked again next time. A confirmation now lasts for the
   session, for writes that can be undone.

2. A 401 used to surface as an httpx sentence repeated on every retry. It now says the
   token needs re-linking, once, in words the user can act on.

3. A list is numbered and one item per line.
"""

from __future__ import annotations

import httpx
import pytest

from app.plugins.github import service
from app.plugins.github.chat_tools import _format_github_result


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            f"Client error '{self.status_code}' for url ...", request=None, response=None
        )


# ---- 2. the token problem says what to do ----------------------------------


def test_a_401_says_to_re_authenticate() -> None:
    with pytest.raises(service.GitHubAuthError) as raised:
        service._raise_github_error(_Response(401))

    message = str(raised.value)
    assert "re-authenticate" in message
    assert "Nothing was read and nothing was changed" in message


def test_a_403_names_the_scope_and_the_rate_limit() -> None:
    with pytest.raises(service.GitHubAuthError) as raised:
        service._raise_github_error(_Response(403))

    assert "scope" in str(raised.value)
    assert "rate limit" in str(raised.value)


def test_a_404_is_not_called_an_auth_failure() -> None:
    with pytest.raises(RuntimeError) as raised:
        service._raise_github_error(_Response(404))

    assert not isinstance(raised.value, service.GitHubAuthError)
    assert "not found" in str(raised.value)


def test_a_good_response_raises_nothing() -> None:
    service._raise_github_error(_Response(200))


def test_a_missing_connection_uses_the_same_sentence(monkeypatch) -> None:
    monkeypatch.setattr(service, "get_integration_token", lambda _user, _integration: None)

    # A ValueError, not a GitHubAuthError: "not connected yet" is a configuration answer
    # the endpoint reports as 400, while a token GitHub rejected is a GitHubAuthError.
    with pytest.raises(ValueError) as raised:
        service._github_token("user-one")

    assert "re-authenticate" in str(raised.value)


# ---- 3. lists are numbered, one item per line ------------------------------


def test_branches_are_numbered_one_per_line() -> None:
    text = _format_github_result(
        "github_list_branches",
        {
            "ok": True,
            "result": {
                "repository": "owner/repo",
                "branches": [
                    {"name": "main", "sha": "a" * 40, "protected": True},
                    {"name": "develop", "sha": "b" * 40, "protected": False},
                ],
            },
        },
    )

    lines = text.splitlines()
    assert lines[1].startswith("1. main")
    assert lines[2].startswith("2. develop")
    assert "- " not in text


def test_repositories_are_numbered_with_their_default_branch() -> None:
    text = _format_github_result(
        "github_list_repositories",
        {
            "ok": True,
            "result": {
                "repositories": [
                    {"full_name": "owner/one", "private": False, "default_branch": "main"},
                    {"full_name": "owner/two", "private": True, "default_branch": "develop"},
                ]
            },
        },
    )

    lines = text.splitlines()
    assert lines[1].startswith("1. owner/one (main)")
    assert lines[2].startswith("2. owner/two (develop)")
