"""What the model is told about connected accounts, and what it is not shown.

Both cases here are the same bug seen twice: the assistant said it had no access to an
account that was connected. Once because nothing in the instruction said an account
existed, and once because its own earlier refusal was still in the history it was sent.
"""

from __future__ import annotations

import pytest

from app.ai.execution import account_context


@pytest.fixture()
def connect(monkeypatch):
    """Connect exactly the named connectors for the tests that need them."""

    def install(connected: set[str]):
        monkeypatch.setattr(
            account_context,
            "get_integration_token_metadata",
            lambda user_id, connector_id: {"integration_id": connector_id} if connector_id in connected else None,
        )

    return install


def test_nothing_is_said_when_nothing_is_connected(connect) -> None:
    connect(set())

    assert account_context.accounts_instruction("u1") == ""


def test_a_connected_account_is_named_and_the_refusal_is_forbidden(connect) -> None:
    connect({"github", "zoho_mail"})

    line = account_context.accounts_instruction("u1")

    assert "github" in line and "zoho_mail" in line
    assert "Never say you cannot access" in line
    assert "call its tool" in line


def test_a_signed_out_request_says_nothing(connect) -> None:
    connect({"github"})

    assert account_context.accounts_instruction("") == ""


def test_the_email_mailbox_is_one_of_the_connectors() -> None:
    """The email tools are gated on a connector that has no tool spec of its own."""
    assert "zoho_mail" in account_context.connector_ids()


def test_an_old_refusal_is_dropped_once_something_is_connected(connect) -> None:
    connect({"github"})
    history = [
        ("user", "how many branches are in my repo"),
        ("assistant", "I don't have access to your GitHub account."),
        ("user", "ok, now list my repositories"),
        ("assistant", "Here are your repositories."),
    ]

    kept = account_context.without_stale_refusals(history, "u1")

    assert kept == [
        ("user", "how many branches are in my repo"),
        ("user", "ok, now list my repositories"),
        ("assistant", "Here are your repositories."),
    ]


def test_the_users_own_words_are_never_dropped(connect) -> None:
    connect({"github"})
    history = [("user", "why did you say you don't have access"), ("assistant", "Sorry.")]

    kept = account_context.without_stale_refusals(history, "u1")

    assert kept == history


def test_a_refusal_is_kept_when_nothing_is_connected(connect) -> None:
    """With nothing connected, the refusal was true and is worth keeping."""
    connect(set())
    history = [("assistant", "I don't have access to your GitHub account.")]

    assert account_context.without_stale_refusals(history, "u1") == history


def test_no_history_stays_none(connect) -> None:
    connect({"github"})

    assert account_context.without_stale_refusals(None, "u1") is None
