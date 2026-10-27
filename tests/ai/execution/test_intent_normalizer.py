"""One reading of a message, shared by the three zero-token routers.

The routers each grew their own patterns, and each one missed a different phrasing - the
same noun with a postposition from another language. These tests hold the normaliser and
the three paths that now use it: a phrasing variation must reach the tool, and a request
that needs judgement must not.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.execution import conversation_digest
from app.ai.execution.intent_normalizer import normalize
from app.plugins.email import email_tools
from app.plugins.github import chat_tools

REPO_LIST = """Most recently updated accessible GitHub repositories:
1. indoone
https://github.com/indooneteam/indoone

2. Indoone-Backend
https://github.com/indooneteam/Indoone-Backend"""

HISTORY = [("user", "list my repos"), ("assistant", REPO_LIST)]


@pytest.mark.parametrize(
    "message",
    [
        "1 alli",
        "1 ralli",
        "1 ge",
        "1 nalli",
        "branches in 1",
        "1 ralli branch list",
    ],
)
def test_an_ordinal_survives_its_postposition(message: str) -> None:
    assert normalize(message).ordinal == 1


@pytest.mark.parametrize(
    ("message", "resource", "ordinal"),
    [
        ("repo 2 pr galu", "pull_requests", 2),
        ("2 ralli branches", "branches", 2),
        ("commits helu", "commits", None),
        ("branch galu torisu", "branches", None),
        ("show commits for 2", "commits", 2),
        ("issues nodu", "issues", None),
        ("3 nalli files", "files", 3),
    ],
)
def test_the_resource_and_the_ordinal_are_read_together(
    message: str, resource: str, ordinal: int | None
) -> None:
    parsed = normalize(message)

    assert parsed.sub_resource == resource
    assert parsed.ordinal == ordinal


@pytest.mark.parametrize("message", ["create a branch", "merge pr 4", "delete the file"])
def test_a_write_verb_is_marked_as_one(message: str) -> None:
    assert normalize(message).is_write is True


@pytest.mark.parametrize(
    "message",
    [
        "write a poem about branches",
        "explain why the build fails",
        "summarise this article",
        "translate this to Hindi",
    ],
)
def test_work_that_needs_judgement_is_marked_leave_alone(message: str) -> None:
    assert normalize(message).leave_alone is True


@pytest.mark.parametrize(
    "message",
    ["1 ralli", "repo 2 pr galu", "branches in 1", "commits helu", "1 ralli branch list"],
)
def test_the_github_router_takes_the_phrasing_variants(message: str) -> None:
    action = chat_tools._github_action(message, HISTORY)

    assert action is not None, f"{message!r} must not reach the model"
    assert isinstance(action, tuple)
    assert action[0].startswith("github_")


def test_a_question_about_branches_is_still_answered_by_the_model() -> None:
    # Names the noun, but asks for reasoning: the fast path must not answer it with a list.
    assert chat_tools._github_action("explain why branches diverge", HISTORY) is None


@pytest.mark.parametrize("message", ["inbox thorisu", "mail helu", "inbox nodu"])
def test_the_email_read_path_takes_the_kanglish_verbs(monkeypatch, message: str) -> None:
    async def fake_chat(*_args, **_kwargs):
        return "1. From: a@b.com — Subject: hello"

    monkeypatch.setattr(email_tools, "handle_email_chat", fake_chat)
    monkeypatch.setattr(email_tools, "get_pending_email_action", lambda *_a, **_k: None)

    assert asyncio.run(email_tools.email_read_fast_path("u1", "c1", message)) is not None


def test_sending_mail_is_not_a_read(monkeypatch) -> None:
    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("a write must not take the read path")

    monkeypatch.setattr(email_tools, "handle_email_chat", must_not_run)
    monkeypatch.setattr(email_tools, "get_pending_email_action", lambda *_a, **_k: None)

    assert asyncio.run(email_tools.email_read_fast_path("u1", "c1", "send a mail to a@b.com")) is None


@pytest.mark.parametrize("message", ["namma maatu helu", "conversation thorisu"])
def test_the_digest_takes_the_kanglish_verbs(monkeypatch, message: str) -> None:
    class FakeStore:
        def recent(self, conversation_id, user_id=None):
            return [("user", "plan the release"), ("assistant", "Created the branch.")]

    monkeypatch.setattr(conversation_digest, "ConversationStore", FakeStore)

    answer = asyncio.run(conversation_digest.conversation_history_answer("u1", "c1", message))

    assert answer is not None
    assert "Topics:" in answer or "ವಿಷಯಗಳು:" in answer


@pytest.mark.parametrize(
    "message",
    ["write a story about my week", "explain why this fails", "what is a monad"],
)
def test_the_digest_leaves_ordinary_talk_alone(monkeypatch, message: str) -> None:
    class FakeStore:
        def recent(self, conversation_id, user_id=None):
            raise AssertionError("the store must not be read for ordinary talk")

    monkeypatch.setattr(conversation_digest, "ConversationStore", FakeStore)

    assert asyncio.run(conversation_digest.conversation_history_answer("u1", "c1", message)) is None
