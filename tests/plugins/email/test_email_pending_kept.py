"""A held email action survives a question, and says plainly that it is still held.

This is the bug behind "it asks for permission, then says it is working on it, and when I
ask whether it is done there is no answer". The pending branch used to end with:

    delete_pending_email_action(user, conversation)

so any message that was not exactly CONFIRM or CANCEL - "did it go?", "wait, what is
happening", "ಏನಾಯ್ತು" - deleted the held action and returned None, which handed the turn
to the model. The model had no idea an action existed, so it answered as if it were
working on one, and the action could never be confirmed again because the record was gone.

Now the record is kept and the answer comes from it.
"""

from __future__ import annotations

import asyncio

import pytest
from cryptography.fernet import Fernet

from app.capabilities.connectors import store
from app.plugins.email import email_assistant


@pytest.fixture(autouse=True)
def _clean_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
    store.initialize()
    yield


def _hold_an_email(monkeypatch) -> None:
    """Put a pending send on the conversation, as prepare_email_to_send does."""

    monkeypatch.setattr(
        email_assistant,
        "get_integration_token_metadata",
        lambda user_id, provider: {"integration_id": "zoho_mail"},
    )
    # The same helper the prepare tools use, so the stored payload is encrypted exactly
    # as it is in production.
    email_assistant._save_pending(
        "user-1",
        "conversation-1",
        "zoho_mail",
        "send_email",
        {"request": "send an email to ravi", "to": "ravi@example.com", "sending": True},
    )


@pytest.mark.parametrize(
    "question",
    ["did it go", "wait what is happening", "ಏನಾಯ್ತು", "क्या हुआ", "is it done yet"],
)
def test_a_question_keeps_the_action_and_says_it_is_waiting(monkeypatch, question: str) -> None:
    _hold_an_email(monkeypatch)

    reply = asyncio.run(email_assistant.handle_email_chat("user-1", "conversation-1", question))

    assert reply is not None, "a question must be answered, not handed to the model"
    assert "still waiting" in reply
    assert store.get_pending_email_action("user-1", "conversation-1") is not None, (
        "the held action must survive a question, or it can never be confirmed"
    )


def test_a_cancel_drops_it(monkeypatch) -> None:
    _hold_an_email(monkeypatch)

    reply = asyncio.run(email_assistant.handle_email_chat("user-1", "conversation-1", "beda"))

    assert reply is not None and "Cancelled" in reply
    assert store.get_pending_email_action("user-1", "conversation-1") is None


def test_a_yes_in_kannada_is_acted_on(monkeypatch) -> None:
    """The confirmation is read in the user's own script, not only in English."""
    _hold_an_email(monkeypatch)
    executed: list[str] = []

    async def fake_execute(user, provider, action_type, data):
        executed.append(action_type)
        return "Email sent."

    monkeypatch.setattr(email_assistant, "_execute_pending", fake_execute)

    reply = asyncio.run(email_assistant.handle_email_chat("user-1", "conversation-1", "ಸರಿ ಕಳಿಸು"))

    assert executed == ["send_email"]
    assert reply == "Email sent."
