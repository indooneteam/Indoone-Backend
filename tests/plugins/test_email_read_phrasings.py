"""A mailbox read that names what it is looking for is still a mailbox read.

"unread emails" was answered deterministically; "mail from boss" and "emails from ravi"
were not, so they reached the model instead of the mailbox.
"""

from __future__ import annotations

import asyncio

import pytest

from app.plugins.email.email_tools import email_read_fast_path


@pytest.mark.parametrize(
    "message",
    [
        "unread emails",
        "show my inbox",
        "check my mail",
        "latest emails",
        "mail from boss",
        "emails from ravi",
        "email about the invoice",
        "emails regarding the contract",
    ],
)
def test_a_read_is_recognised(message: str) -> None:
    assert asyncio.run(email_read_fast_path("u1", "c1", message)) is not None


@pytest.mark.parametrize(
    "message",
    [
        "send a mail to ravi",
        "reply to that email",
        "delete this email",
        "hi",
        "when is deepavali",
        "list branches",
    ],
)
def test_a_write_and_another_subject_are_left_alone(message: str) -> None:
    assert asyncio.run(email_read_fast_path("u1", "c1", message)) is None
