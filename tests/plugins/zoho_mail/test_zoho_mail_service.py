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
