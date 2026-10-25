from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from app.plugins.email import email_assistant
def test_non_email_chat_is_not_intercepted() -> None:
    import asyncio
    result = asyncio.run(email_assistant.handle_email_chat("user", "conversation", "Explain the moon phases"))
    assert result is None


@pytest.mark.asyncio
async def test_email_request_without_connection_is_explained(monkeypatch) -> None:
    monkeypatch.setattr(email_assistant, "get_integration_token_metadata", lambda user_id, integration_id: None)
    result = await email_assistant.handle_email_chat("user", "conversation", "Do I have any recent emails?")
    assert result is not None
    assert "connect" in result.casefold()


@pytest.mark.asyncio
async def test_zoho_finance_search_uses_connected_account(monkeypatch) -> None:
    monkeypatch.setattr(
        email_assistant,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {"user_id": user_id} if integration_id == "zoho_mail" else None,
    )

    async def fake_accounts(user_id):
        return {"data": [{"accountId": "account-1", "primaryEmailAddress": "me@zoho.example"}]}

    async def fake_messages(user_id, account_id, **kwargs):
        assert account_id == "account-1"
        assert kwargs["search_key"] == "finance"
        return {"data": [{
            "messageId": "message-1",
            "folderId": "folder-1",
            "subject": "Finance report",
            "sender": "finance@example.com",
            "summary": "Monthly report attached",
        }]}

    monkeypatch.setattr(email_assistant, "list_accounts", fake_accounts)
    monkeypatch.setattr(email_assistant, "list_messages", fake_messages)

    result = await email_assistant.handle_email_chat(
        "user", "conversation", "Check Finance emails in Zoho Mail"
    )
    assert result is not None
    assert "Finance report" in result
    assert "finance@example.com" in result


@pytest.mark.asyncio
async def test_delete_requires_confirmation_and_uses_scoped_pending_action(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setattr(
        email_assistant,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {"user_id": user_id} if integration_id == "zoho_mail" else None,
    )

    async def fake_accounts(user_id):
        return {"data": [{"accountId": "account-1", "primaryEmailAddress": "me@zoho.example"}]}

    async def fake_messages(user_id, account_id, **kwargs):
        return {"data": [{
            "messageId": "message-1",
            "folderId": "folder-1",
            "subject": "Finance report",
            "sender": "finance@example.com",
            "summary": "Monthly report",
        }]}

    deleted = []

    async def fake_delete(user_id, account_id, folder_id, message_id, **kwargs):
        deleted.append((user_id, account_id, folder_id, message_id, kwargs.get("approved")))
        return {"secrets_exposed": False}

    monkeypatch.setattr(email_assistant, "list_accounts", fake_accounts)
    monkeypatch.setattr(email_assistant, "list_messages", fake_messages)
    monkeypatch.setattr(email_assistant, "delete_message", fake_delete)

    first = await email_assistant.handle_email_chat(
        "firebase-user-1", "conversation-1", "Delete Finance emails in Zoho Mail"
    )
    assert first is not None
    assert "CONFIRM" in first
    assert deleted == []

    result = await email_assistant.handle_email_chat("firebase-user-1", "conversation-1", "CONFIRM")
    assert result is not None
    assert "Moved 1 email" in result
    assert deleted == [("firebase-user-1", "account-1", "folder-1", "message-1", True)]


@pytest.mark.asyncio
async def test_send_request_only_sends_after_confirmation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setattr(
        email_assistant,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {"user_id": user_id} if integration_id == "zoho_mail" else None,
    )

    async def fake_compose(request, recipient):
        assert recipient == "friend@example.com"
        return "Hello friend", "Hi friend, hope you are doing well."

    sent = []

    async def fake_send(user_id, account_id, from_address, to, subject, body, approved=False):
        sent.append((user_id, to, subject, body, approved))
        return {
            "status": {"code": "200"},
            "data": {"messageId": "provider-message-id"},
            "secrets_exposed": False,
        }

    async def fake_list_accounts(user_id):
        return {"data": [{"accountId": "acct-1", "primaryEmailAddress": "sender@example.com"}]}

    monkeypatch.setattr(email_assistant, "list_accounts", fake_list_accounts)
    monkeypatch.setattr(email_assistant, "_compose_fields", fake_compose)
    monkeypatch.setattr(email_assistant, "send_message", fake_send)

    preview = await email_assistant.handle_email_chat(
        "firebase-user-2",
        "conversation-2",
        "Send a friendly email to friend@example.com",
    )
    assert preview is not None
    assert "CONFIRM" in preview
    assert sent == []

    result = await email_assistant.handle_email_chat("firebase-user-2", "conversation-2", "yes")
    assert result is not None
    assert "sent successfully" in result.casefold()
    assert sent == [(
        "firebase-user-2",
        "friend@example.com",
        "Hello friend",
        "Hi friend, hope you are doing well.",
        True,
    )]


def test_sender_word_does_not_trigger_send_intent() -> None:
    assert email_assistant._is_send(
        "List my recent Zoho Mail emails and show sender, subject and date"
    ) is False
    assert email_assistant._is_send("Send an email to friend@example.com") is True
    assert email_assistant._is_send("I'm sending an email to friend@example.com") is True

