from __future__ import annotations

import asyncio

import pytest

from app.plugins import email_assistant, email_tools


def test_resolve_provider_requires_choice_when_both_mailboxes_connected(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider},
    )

    provider, error = email_tools._resolve_provider("user-1", None)

    assert provider is None
    assert error is not None
    assert error["code"] == "PROVIDER_REQUIRED"


def test_resolve_provider_rejects_unconnected_explicit_mailbox(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "gmail" else None,
    )

    provider, error = email_tools._resolve_provider("user-1", "zoho_mail")

    assert provider is None
    assert error is not None
    assert error["code"] == "NOT_CONNECTED"


def test_prepare_send_only_stores_pending_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider} if provider == "gmail" else None,
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_to_send",
            {
                "provider": "gmail",
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
        lambda user_id, provider: {"provider": provider} if provider == "gmail" else None,
    )
    monkeypatch.setattr(
        email_tools,
        "_list_mail",
        lambda *args, **kwargs: _async_value([
            {"id": "message-1", "subject": "Finance invoice", "from": "finance@example.com"},
            {"id": "message-2", "subject": "Finance report", "from": "finance@example.com"},
        ]),
    )
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_deletion",
            {"provider": "gmail", "query": "Finance", "limit": 5},
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
        lambda user_id, provider: {"provider": provider} if provider == "gmail" else None,
    )

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_deletion",
            {"provider": "gmail", "query": "all emails"},
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
    saved: list[tuple[object, ...]] = []
    monkeypatch.setattr(email_tools, "_save_pending", lambda *args: saved.append(args))

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "prepare_email_to_send",
            {
                "provider": "gmail",
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
            {"provider": "gmail", "limit": 1},
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
        lambda user_id, provider: {"provider": provider} if provider == "gmail" else None,
    )
    created: list[tuple[str, str, str]] = []

    async def fake_create_draft(user_id: str, to: str = "", subject: str = "", body: str = ""):
        created.append((to, subject, body))
        return {"operation": "draft_saved"}

    monkeypatch.setattr(email_tools, "create_gmail_draft", fake_create_draft)

    async def fake_model(message, *, history, tool_declarations, execute_tool, extra_system_instruction):
        first = await execute_tool(
            "create_email_draft",
            {"provider": "gmail", "to": "a@example.com", "subject": "First", "body": "First body"},
        )
        second = await execute_tool(
            "create_email_draft",
            {"provider": "gmail", "to": "b@example.com", "subject": "Second", "body": "Second body"},
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



def test_model_cannot_guess_between_two_connected_mailboxes(monkeypatch) -> None:
    monkeypatch.setattr(
        email_tools,
        "get_integration_token_metadata",
        lambda user_id, provider: {"provider": provider},
    )

    result = asyncio.run(
        email_tools.execute_email_tool(
            "user-1",
            "conversation-1",
            "list_recent_emails",
            {"provider": "gmail"},
            original_message="Check my email",
        )
    )

    assert result["ok"] is False
    assert result["code"] == "PROVIDER_REQUIRED"
