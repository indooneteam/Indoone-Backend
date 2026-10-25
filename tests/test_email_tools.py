from __future__ import annotations

import asyncio
import app.plugins.email_assistant as email_assistant
from cryptography.fernet import Fernet

import pytest

from app.plugins import email_assistant, email_tools


def test_resolve_provider_rejects_unconnected_explicit_mailbox(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: None,
    )

    provider, error = email_tools._resolve_provider("user-1", "zoho_mail")

    assert provider is None
    assert error is not None
    assert error["code"] == "NOT_CONNECTED"


def test_prepare_send_only_stores_pending_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "list_accounts",
        lambda user_id: _async_value({"data": [{"accountId": "acct-1", "primaryEmailAddress": "sender@example.com"}]}),
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_to_send",
            {
                "provider": "zoho_mail",
                "to": "customer@example.com",
                "subject": "Invoice update",
                "body": "The invoice is ready.",
            },
            original_message="Please send this email to customer@example.com",
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "awaiting_confirmation"
    assert result["sent"] is False
    assert len(saved) == 1
    assert saved[0][3] == "send_email"
    assert saved[0][4]["to"] == "customer@example.com"


def test_prepare_delete_only_stores_verified_targets_and_never_deletes(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_list_mail",
        lambda *args, **kwargs: _async_value([
            {"id": "message-1", "subject": "Finance invoice", "from": "finance@example.com", "account_id": "acct-1", "folder_id": "inbox-1"},
            {"id": "message-2", "subject": "Finance report", "from": "finance@example.com", "account_id": "acct-1", "folder_id": "inbox-1"},
        ]),
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_deletion",
            {"provider": "zoho_mail", "query": "Finance", "limit": 5},
            original_message="Delete matching Finance emails",
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "awaiting_confirmation"
    assert result["deleted"] is False
    assert result["count"] == 2
    assert len(saved) == 1
    assert saved[0][3] == "delete_messages"
    assert [item["id"] for item in saved[0][4]["items"]] == ["message-1", "message-2"]


async def _async_value(value):
    return value


def test_prepare_delete_rejects_unfiltered_delete_request(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_deletion",
            {"provider": "zoho_mail", "query": "all emails"},
            original_message="Delete all emails",
        )
    )

    assert result["ok"] is False
    assert result["code"] == "FILTER_REQUIRED"


def test_zoho_recent_mail_uses_inbox_folder_and_received_time_order(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_list_accounts(user_id: str) -> dict[str, object]:
        return {}

    async def fake_list_folders(user_id: str, account_id: str) -> dict[str, object]:
        return {"data": [{"folderName": "Inbox", "folderId": "inbox-123"}]}

    async def fake_list_messages(
        user_id: str,
        account_id: str,
        *,
        folder_id: str = "",
        search_key: str = "",
        start: int = 1,
        limit: int = 20,
    ) -> dict[str, object]:
        calls.append({"folder_id": folder_id, "search_key": search_key, "start": start, "limit": limit})
        return {
            "data": [
                {"messageId": "older", "folderId": folder_id, "subject": "Older", "receivedTime": "2026-10-01T09:00:00Z"},
                {"messageId": "newer", "folderId": folder_id, "subject": "Newer", "receivedTime": "2026-10-10T09:00:00Z"},
            ]
        }

    monkeypatch.setattr(email_assistant, "list_accounts", fake_list_accounts)
    monkeypatch.setattr(email_assistant, "list_folders", fake_list_folders)
    monkeypatch.setattr(email_assistant, "list_messages", fake_list_messages)
    monkeypatch.setattr(
        email_assistant,
        "_zoho_accounts",
        lambda payload: [{"id": "account-123", "email": "me@example.com"}],
    )

    rows = asyncio.run(email_assistant._list_mail("user-1", "zoho_mail", "", recent=True, limit=7))

    assert calls == [{"folder_id": "inbox-123", "search_key": "", "start": 1, "limit": 7}]
    assert [row["id"] for row in rows] == ["newer", "older"]
    assert all(row["folder_id"] == "inbox-123" for row in rows)


def test_zoho_recent_mail_fails_closed_if_inbox_folder_is_unavailable(monkeypatch) -> None:
    async def fake_list_accounts(user_id: str) -> dict[str, object]:
        return {}

    async def fake_list_folders(user_id: str, account_id: str) -> dict[str, object]:
        return {"data": [{"folderName": "Sent", "folderId": "sent-1"}]}

    monkeypatch.setattr(email_assistant, "list_accounts", fake_list_accounts)
    monkeypatch.setattr(email_assistant, "list_folders", fake_list_folders)
    monkeypatch.setattr(
        email_assistant,
        "_zoho_accounts",
        lambda payload: [{"id": "account-123", "email": "me@example.com"}],
    )

    with pytest.raises(RuntimeError, match="Inbox folder could not be identified"):
        asyncio.run(email_assistant._list_mail("user-1", "zoho_mail", "", recent=True))



def test_confirmation_message_is_processed_even_without_email_keyword(monkeypatch) -> None:
    called: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        email_tools,
        "get_pending_email_action",
        lambda user_id, conversation_id: {"action_type": "send_email"},
    )

    async def fake_handle(user_id: str, conversation_id: str, message: str) -> str:
        called.append((user_id, conversation_id, message))
        return "Email sent successfully."

    monkeypatch.setattr(email_tools, "handle_email_chat", fake_handle)

    result = asyncio.run(
        email_tools.handle_email_chat_with_tools(
            "user-1",
            "conversation-1",
            "CONFIRM",
            history=None,
        )
    )

    assert result == "Email sent successfully."
    assert called == [("user-1", "conversation-1", "CONFIRM")]



def test_send_tool_is_rejected_without_explicit_send_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider},
    )
    monkeypatch.setattr(
        email_tools,
        "list_accounts",
        lambda user_id: _async_value({"data": [{"accountId": "acct-1", "primaryEmailAddress": "sender@example.com"}]}),
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_to_send",
            {
                "provider": "zoho_mail",
                "to": "customer@example.com",
                "subject": "Unexpected",
                "body": "This must not be sent.",
            },
            original_message="Check my emails for the latest invoice",
        )
    )

    assert result["ok"] is False
    assert result["code"] == "ACTION_NOT_REQUESTED"
    assert saved == []



def test_tool_uses_mailbox_named_by_user_over_model_argument(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_list_mail",
        lambda *args, **kwargs: _async_value([
            {"id": "zoho-message", "subject": "Zoho result", "from": "sender@example.com"}
        ]),
    )

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "list_recent_emails",
            {"provider": "zoho_mail", "limit": 1},
            original_message="Show recent Zoho Mail",
        )
    )

    assert result["ok"] is True
    assert result["provider"] == "Zoho Mail"
    assert result["emails"][0]["id"] == "zoho-message"



def test_only_one_write_operation_is_allowed_in_a_single_chat_turn(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setattr(email_tools, "get_pending_email_action", lambda user_id, conversation_id: None)
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    created: list[tuple[str, str, str]] = []

    async def fake_list_accounts(user_id: str):
        return {"data": [{"accountId": "acct-1", "primaryEmailAddress": "sender@example.com"}]}

    async def fake_save_draft(user_id: str, account_id: str, from_address: str, to: str = "", subject: str = "", body: str = ""):
        created.append((to, subject, body))
        return {"operation": "draft_saved"}

    monkeypatch.setattr(email_tools, "list_accounts", fake_list_accounts)
    monkeypatch.setattr(email_tools, "save_draft", fake_save_draft)

    async def fake_model(message, *, history, tool_declarations, execute_tool, extra_system_instruction):
        first = await execute_tool(
            "create_email_draft",
            {"provider": "zoho_mail", "to": "a@example.com", "subject": "First", "body": "First body"},
        )
        second = await execute_tool(
            "create_email_draft",
            {"provider": "zoho_mail", "to": "b@example.com", "subject": "Second", "body": "Second body"},
        )
        assert first["ok"] is True
        assert second["code"] == "MULTIPLE_WRITES_BLOCKED"
        return "A single draft was saved."

    import app.ai.gemini_service

    monkeypatch.setattr(app.ai.gemini_service, "generate_gemini_function_call_reply", fake_model)

    result = asyncio.run(
        email_tools.handle_email_chat_with_tools(
            "user-1",
            "conversation-1",
            "Please draft an email for a@example.com",
            history=None,
        )
    )

    assert result == "A single draft was saved."
    assert created == [("a@example.com", "First", "First body")]



def test_send_preparation_cannot_be_reported_as_sent_by_model(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setattr(email_tools, "get_pending_email_action", lambda user_id, conversation_id: None)
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_save_pending",
        lambda *args: None,
    )

    async def fake_list_accounts(user_id: str) -> dict[str, object]:
        return {"data": [{"accountId": "account-1", "primaryEmailAddress": "sender@example.com"}]}

    monkeypatch.setattr(email_tools, "list_accounts", fake_list_accounts)

    async def fake_model(message, *, history, tool_declarations, execute_tool, extra_system_instruction):
        result = await execute_tool(
            "prepare_email_to_send",
            {
                "provider": "zoho_mail",
                "to": "customer@example.com",
                "subject": "Test subject",
                "body": "Test body",
            },
        )
        assert result["operation"] == "awaiting_confirmation"
        assert result["sent"] is False
        # Simulate the model's exact bug: claiming success before confirmation.
        return "Email sent successfully to customer@example.com."

    import app.ai.gemini_service

    monkeypatch.setattr(app.ai.gemini_service, "generate_gemini_function_call_reply", fake_model)

    result = asyncio.run(
        email_tools.handle_email_chat_with_tools(
            "user-1",
            "conversation-1",
            "Send a Zoho Mail email to customer@example.com",
            history=None,
        )
    )

    assert "NOT been sent" in result
    assert "CONFIRM" in result
    assert "Email sent successfully" not in result
    assert "customer@example.com" in result
    assert "Test subject" in result
    assert "Test body" in result


def test_zoho_send_is_not_reported_successful_without_provider_message_id(monkeypatch) -> None:
    async def fake_send_message(*args, **kwargs):
        return {
            "status": {"code": 200, "description": "success"},
            "data": {"subject": "Test subject"},
        }

    monkeypatch.setattr(email_assistant, "send_message", fake_send_message)

    with pytest.raises(RuntimeError, match="message ID"):
        asyncio.run(
            email_assistant._execute_pending(
                "user-1",
                "zoho_mail",
                "send_email",
                {
                    "account_id": "account-1",
                    "from_address": "sender@example.com",
                    "to": "customer@example.com",
                    "subject": "Test subject",
                    "body": "Test body",
                },
            )
        )


def test_verified_zoho_send_can_be_reported_successful(monkeypatch) -> None:
    async def fake_send_message(*args, **kwargs):
        return {
            "status": {"code": 200, "description": "success"},
            "data": {"messageId": "provider-message-id"},
        }

    monkeypatch.setattr(email_assistant, "send_message", fake_send_message)

    result = asyncio.run(
        email_assistant._execute_pending(
            "user-1",
            "zoho_mail",
            "send_email",
            {
                "account_id": "account-1",
                "from_address": "sender@example.com",
                "to": "customer@example.com",
                "subject": "Test subject",
                "body": "Test body",
            },
        )
    )

    assert result == "Email sent successfully through Zoho Mail."

def test_prepare_zoho_reply_requires_confirmation_and_resolves_recipient(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_list_mail",
        lambda *args, **kwargs: _async_value([
            {
                "id": "message-123",
                "account_id": "account-1",
                "folder_id": "inbox-1",
                "subject": "Launch plan",
                "from": "Alice Example <alice@example.com>",
            }
        ]),
    )
    monkeypatch.setattr(email_tools, "_zoho_accounts", lambda payload: [{"id": "account-1", "email": "owner@zoho.example"}])
    monkeypatch.setattr(email_tools, "list_accounts", lambda user_id: _async_value({}))
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_reply",
            {"provider": "zoho_mail", "query": "Launch plan", "body": "The revised date is Friday."},
            original_message="Reply to the Zoho Mail email about Launch plan with the revised date",
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "awaiting_confirmation"
    assert result["sent"] is False
    assert result["to"] == "alice@example.com"
    assert result["subject"] == "Re: Launch plan"
    assert len(saved) == 1
    assert saved[0][3] == "send_reply"
    assert saved[0][4]["message_id"] == "message-123"
    assert saved[0][4]["from_address"] == "owner@zoho.example"
    assert saved[0][4]["to"] == "alice@example.com"


def test_prepare_zoho_draft_deletion_stores_only_confirmed_target(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_list_drafts",
        lambda *args, **kwargs: _async_value([
            {
                "id": "draft-123",
                "draft_id": "draft-123",
                "account_id": "account-1",
                "folder_id": "drafts-1",
                "subject": "Quarterly update",
                "snippet": "Share the finance summary",
            }
        ]),
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_draft_deletion",
            {"provider": "zoho_mail", "query": "Quarterly update"},
            original_message="Delete the Zoho Mail draft named Quarterly update",
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "awaiting_confirmation"
    assert result["deleted"] is False
    assert result["count"] == 1
    assert len(saved) == 1
    assert saved[0][3] == "delete_drafts"
    assert saved[0][4]["items"] == [
        {"draft_id": "draft-123", "id": "draft-123", "account_id": "account-1", "folder_id": "drafts-1"}
    ]


def test_zoho_reply_confirmation_requires_verifiable_message_id(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_reply_to_message(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"status": {"code": 200, "description": "success"}, "data": {"subject": "Re: Launch plan"}}

    monkeypatch.setattr(email_assistant, "reply_to_message", fake_reply_to_message)

    with pytest.raises(RuntimeError, match="message ID"):
        asyncio.run(
            email_assistant._execute_pending(
                "user-1",
                "zoho_mail",
                "send_reply",
                {
                    "account_id": "account-1",
                    "message_id": "message-123",
                    "from_address": "owner@zoho.example",
                    "to": "alice@example.com",
                    "subject": "Re: Launch plan",
                    "body": "The revised date is Friday.",
                },
            )
        )

    assert calls[0]["kwargs"]["approved"] is True

def test_zoho_summary_prefers_actual_sender_address_and_provider_timestamps() -> None:
    result = email_assistant._zoho_summary(
        {
            "id": "message-1",
            "sender": "Alice Example",
            "fromAddress": "alice@example.com",
            "subject": "Launch plan",
            "receivedtime": "2026-10-11T09:30:00Z",
            "folderId": "inbox-1",
            "toAddress": "owner@zoho.example",
        },
        account_id="account-1",
    )

    assert result["from"] == "alice@example.com"
    assert result["date"] == "2026-10-11T09:30:00Z"
    assert result["account_id"] == "account-1"
    assert result["folder_id"] == "inbox-1"
    assert result["to"] == "owner@zoho.example"

def test_explicit_reply_intent_rejects_negated_english_and_kannada_requests() -> None:
    for message in (
        "Don't reply to that email",
        "Do not send a reply to this message",
        "ರಿಪ್ಲೈ ಬೇಡ",
        "ರಿಪ್ಲೈ ಮಾಡಬೇಡ",
        "ಉತ್ತರ ಕಳುಹಿಸಬೇಡಿ",
    ):
        assert email_tools._explicit_reply_intent(message) is False


def test_explicit_reply_intent_accepts_a_direct_reply_request() -> None:
    assert email_tools._explicit_reply_intent("Reply to the Zoho Mail message about Launch plan") is True


def test_zoho_summary_formats_epoch_milliseconds_as_utc() -> None:
    result = email_assistant._zoho_summary(
        {
            "messageId": "message-2",
            "fromAddress": "sender@example.com",
            "subject": "Quarterly report",
            "receivedTime": "1709887053409",
            "folderId": "inbox-1",
        },
        account_id="account-1",
    )

    assert result["date"] == "2024-03-08T08:37:33.409Z"

def test_verified_zoho_reply_can_be_reported_successful(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_reply_to_message(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {
            "status": {"code": 200, "description": "success"},
            "data": {"messageId": "provider-reply-id"},
        }

    monkeypatch.setattr(email_assistant, "reply_to_message", fake_reply_to_message)
    result = asyncio.run(
        email_assistant._execute_pending(
            "user-1",
            "zoho_mail",
            "send_reply",
            {
                "account_id": "account-1",
                "message_id": "original-message-id",
                "from_address": "owner@zoho.example",
                "to": "alice@example.com",
                "subject": "Re: Launch plan",
                "body": "The revised date is Friday.",
            },
        )
    )

    assert result == "Reply sent successfully through Zoho Mail."
    assert len(calls) == 1
    assert calls[0]["kwargs"]["approved"] is True
    assert calls[0]["args"][2] == "original-message-id"
    assert calls[0]["args"][4] == "alice@example.com"


def test_zoho_reply_rejects_provider_failure_status(monkeypatch) -> None:
    async def fake_reply_to_message(*args, **kwargs):
        return {
            "status": {"code": 400, "description": "invalid target"},
            "data": {"messageId": "not-a-success"},
        }

    monkeypatch.setattr(email_assistant, "reply_to_message", fake_reply_to_message)

    with pytest.raises(RuntimeError, match="verified successful reply status"):
        asyncio.run(
            email_assistant._execute_pending(
                "user-1",
                "zoho_mail",
                "send_reply",
                {
                    "account_id": "account-1",
                    "message_id": "original-message-id",
                    "from_address": "owner@zoho.example",
                    "to": "alice@example.com",
                    "subject": "Re: Launch plan",
                    "body": "The revised date is Friday.",
                },
            )
        )



def test_email_tool_fetch_limit_defaults_to_200_and_caps_at_200() -> None:
    assert email_tools._limit(None) == 200
    assert email_tools._limit(25) == 25
    assert email_tools._limit(200) == 200
    assert email_tools._limit(999) == 200
    assert email_tools._limit(None, default=20, maximum=20) == 20


def test_zoho_inbox_count_paginates_without_counting_only_the_first_page(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_accounts(user_id: str) -> dict[str, object]:
        return {"data": [{"accountId": "account-1", "primaryEmailAddress": "me@example.com"}]}

    async def fake_folders(user_id: str, account_id: str) -> dict[str, object]:
        return {"data": [{"folderName": "Inbox", "folderId": "inbox-1"}]}

    async def fake_messages(
        user_id: str,
        account_id: str,
        *,
        folder_id: str = "",
        search_key: str = "",
        start: int = 1,
        limit: int = 20,
    ) -> dict[str, object]:
        calls.append({"start": start, "limit": limit, "folder_id": folder_id})
        size = 200 if start == 1 else 37
        return {"data": [{"messageId": f"m-{start + i}"} for i in range(size)]}

    monkeypatch.setattr(email_tools, "list_accounts", fake_accounts)
    monkeypatch.setattr(email_tools, "list_folders", fake_folders)
    monkeypatch.setattr(email_tools, "list_messages", fake_messages)
    monkeypatch.setattr(
        email_tools, "_zoho_accounts",
        lambda _payload: [{"id": "account-1", "email": "me@example.com"}],
    )

    result = asyncio.run(email_tools._count_inbox_emails("user-1", "zoho_mail"))

    assert result == {"folder": "Inbox", "count": 237, "exact": True}
    assert calls == [
        {"start": 1, "limit": 200, "folder_id": "inbox-1"},
        {"start": 201, "limit": 200, "folder_id": "inbox-1"},
    ]


def test_count_inbox_emails_is_an_explicit_live_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "zoho_mail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_count_inbox_emails",
        lambda user_id, provider: _async_value({"folder": "Inbox", "count": 237, "exact": True}),
    )

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "count_inbox_emails",
            {"provider": "zoho_mail"},
            original_message="How many emails are in Zoho Mail Inbox?",
        )
    )

    assert result["ok"] is True
    assert result["count"] == 237
    assert result["exact"] is True


def test_zoho_recent_fetch_uses_provider_maximum_of_200(monkeypatch) -> None:
    calls: list[int] = []

    async def fake_accounts(user_id: str) -> dict[str, object]:
        return {"data": [{"accountId": "account-1", "primaryEmailAddress": "me@example.com"}]}

    async def fake_folders(user_id: str, account_id: str) -> dict[str, object]:
        return {"data": [{"folderName": "Inbox", "folderId": "inbox-1"}]}

    async def fake_messages(
        user_id: str,
        account_id: str,
        *,
        folder_id: str = "",
        search_key: str = "",
        start: int = 1,
        limit: int = 20,
    ) -> dict[str, object]:
        calls.append(limit)
        return {"data": [{"messageId": "m-1", "folderId": folder_id, "subject": "Hello"}]}

    monkeypatch.setattr(email_assistant, "list_accounts", fake_accounts)
    monkeypatch.setattr(email_assistant, "list_folders", fake_folders)
    monkeypatch.setattr(email_assistant, "list_messages", fake_messages)
    monkeypatch.setattr(
        email_assistant, "_zoho_accounts",
        lambda _payload: [{"id": "account-1", "email": "me@example.com"}],
    )

    rows = asyncio.run(
        email_assistant._list_mail("user-1", "zoho_mail", "", recent=True, limit=200)
    )

    assert len(rows) == 1
    assert calls == [200]


def test_confirming_a_zoho_send_reaches_the_provider(monkeypatch, tmp_path) -> None:
    """End to end: save a pending send, confirm it, and prove the provider was called.

    This is the link nothing covered. One test proved the request saves a pending action,
    another proved a bare "yes" reaches handle_email_chat, but nothing proved that the
    confirm actually reaches Zoho Mail. That is the link that broke in production.
    """
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())

    email_tools._save_pending(
        "user-1",
        "conversation-1",
        "zoho_mail",
        "send_email",
        {
            "account_id": "111",
            "from_address": "me@example.com",
            "to": "ravi@example.com",
            "subject": "Hello",
            "body": "Body text",
        },
    )

    sent: list[dict[str, object]] = []

    async def fake_send_message(
        user_id, account_id, from_address, to_address, subject, content, approved=False
    ):
        sent.append(
            {
                "user_id": user_id,
                "account_id": account_id,
                "from": from_address,
                "to": to_address,
                "subject": subject,
                "content": content,
                "approved": approved,
            }
        )
        return {"status": {"code": 200}, "data": {"messageId": "msg-1"}}

    monkeypatch.setattr(email_assistant, "send_message", fake_send_message)

    reply = asyncio.run(email_assistant.handle_email_chat("user-1", "conversation-1", "yes"))

    assert len(sent) == 1, "the confirmed send never reached the provider"
    assert sent[0]["to"] == "ravi@example.com"
    assert sent[0]["subject"] == "Hello"
    assert sent[0]["approved"] is True, "the send must be explicitly approved"
    assert "sent" in reply.lower()
