from __future__ import annotations

import asyncio

import pytest

from app.plugins.zoho_mail import service


def test_replace_draft_keeps_original_when_new_draft_id_is_not_verifiable(monkeypatch) -> None:
    async def fake_save_draft(*args, **kwargs):
        return {"status": {"code": 200, "description": "success"}, "data": {"subject": "Revised"}}

    deleted: list[tuple[object, ...]] = []

    async def fake_delete_message(*args, **kwargs):
        deleted.append(args)
        return {"status": {"code": 200}}

    monkeypatch.setattr(service, "save_draft", fake_save_draft)
    monkeypatch.setattr(service, "delete_message", fake_delete_message)

    with pytest.raises(RuntimeError, match="original draft was kept"):
        asyncio.run(
            service.replace_draft(
                "user-1",
                "account-1",
                "drafts-1",
                "old-draft",
                "owner@example.com",
                "reader@example.com",
                "Revised subject",
                "Revised body",
                approved=True,
            )
        )

    assert deleted == []


def test_replace_draft_verifies_new_id_before_removing_original(monkeypatch) -> None:
    async def fake_save_draft(*args, **kwargs):
        return {
            "status": {"code": 200, "description": "success"},
            "data": {"messageId": "new-draft"},
        }

    deleted: list[tuple[object, ...]] = []

    async def fake_delete_message(*args, **kwargs):
        deleted.append(args)
        return {"status": {"code": 200}}

    monkeypatch.setattr(service, "save_draft", fake_save_draft)
    monkeypatch.setattr(service, "delete_message", fake_delete_message)

    result = asyncio.run(
        service.replace_draft(
            "user-1",
            "account-1",
            "drafts-1",
            "old-draft",
            "owner@example.com",
            "reader@example.com",
            "Revised subject",
            "Revised body",
            approved=True,
        )
    )

    assert result["new_draft_id"] == "new-draft"
    assert result["replaced_message_id"] == "old-draft"
    assert len(deleted) == 1
    assert deleted[0][-1] == "old-draft"

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("launch planning", 'entire:"launch planning"'),
        ("from:alice@example.com", "sender:alice@example.com"),
        ("folder:Sent Items", 'in:"Sent Items"'),
        ("tag:finance", "label:finance"),
        ("before:2026-10-12", "toDate:12-Oct-2026"),
        ("after:12-Oct-2026", "fromDate:12-Oct-2026"),
        ("subject:invoice::has:attachment", "subject:invoice::has:attachment"),
        (
            "sender:alice@example.com::or:to:alice@example.com",
            "sender:alice@example.com::or:to:alice@example.com",
        ),
        (
            "on:2026-10-12",
            "fromDate:12-Oct-2026::toDate:12-Oct-2026",
        ),
    ],
)
def test_list_messages_normalizes_search_key_to_zoho_syntax(monkeypatch, query, expected) -> None:
    captured: dict[str, object] = {}

    async def fake_request(user_id, method, path, *, params=None, payload=None):
        captured.update({"user_id": user_id, "method": method, "path": path, "params": params})
        return {"status": {"code": 200, "description": "success"}, "data": []}

    monkeypatch.setattr(service, "_request", fake_request)
    asyncio.run(service.list_messages("user-1", "account-1", search_key=query))

    assert captured["path"] == "accounts/account-1/messages/search"
    assert captured["params"]["searchKey"] == expected


def test_list_messages_rejects_unsupported_bcc_search_filter(monkeypatch) -> None:
    async def fake_request(*args, **kwargs):
        raise AssertionError("Invalid query must be rejected before making an API request")

    monkeypatch.setattr(service, "_request", fake_request)
    with pytest.raises(ValueError, match="does not support bcc"):
        asyncio.run(service.list_messages("user-1", "account-1", search_key="bcc:private@example.com"))

