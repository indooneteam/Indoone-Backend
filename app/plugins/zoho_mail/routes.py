"""Zoho Mail account and message operations."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.zoho_mail.service import (
    get_message_content,
    list_accounts,
    list_folders,
    delete_message,
    list_messages,
    replace_draft,
    reply_to_message,
    save_draft,
    send_message,
)

router = APIRouter(prefix="/integrations/zoho-mail", tags=["zoho-mail"])


class ZohoMailUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class ZohoMailAccountRequest(ZohoMailUserRequest):
    account_id: str = Field(min_length=1, max_length=256)


class ZohoMailMessagesRequest(ZohoMailAccountRequest):
    folder_id: str = Field(default="", max_length=256)
    search_key: str = Field(default="", max_length=512)
    start: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=200)


class ZohoMailMessageRequest(ZohoMailAccountRequest):
    folder_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)


class ZohoMailSendRequest(ZohoMailAccountRequest):
    from_address: str = Field(min_length=3, max_length=512)
    to_address: str = Field(min_length=3, max_length=512)
    subject: str = Field(min_length=1, max_length=2000)
    content: str = Field(min_length=1, max_length=20_000)
    approved: bool = False

class ZohoMailReplyRequest(ZohoMailAccountRequest):
    message_id: str = Field(min_length=1, max_length=256)
    from_address: str = Field(min_length=3, max_length=512)
    to_address: str = Field(min_length=3, max_length=512)
    subject: str = Field(min_length=1, max_length=2000)
    content: str = Field(min_length=1, max_length=20_000)
    approved: bool = False


class ZohoMailDeleteRequest(ZohoMailAccountRequest):
    folder_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    approved: bool = False
    permanently: bool = False


class ZohoMailDraftRequest(ZohoMailAccountRequest):
    from_address: str = Field(min_length=3, max_length=512)
    to_address: str = Field(default="", max_length=512)
    subject: str = Field(default="", max_length=2000)
    content: str = Field(default="", max_length=20_000)


class ZohoMailReplaceDraftRequest(ZohoMailDraftRequest):
    folder_id: str = Field(min_length=1, max_length=256)
    draft_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


def _raise_provider_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, httpx.HTTPError):
        raise HTTPException(status_code=502, detail="Zoho Mail API request failed") from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail="Zoho Mail request failed") from exc


@router.post("/accounts")
async def zoho_mail_accounts(request: ZohoMailUserRequest) -> dict[str, object]:
    try:
        return await list_accounts(request.user_id)
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/folders")
async def zoho_mail_folders(request: ZohoMailAccountRequest) -> dict[str, object]:
    try:
        return await list_folders(request.user_id, request.account_id)
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/messages")
async def zoho_mail_messages(request: ZohoMailMessagesRequest) -> dict[str, object]:
    try:
        return await list_messages(
            request.user_id,
            request.account_id,
            folder_id=request.folder_id,
            search_key=request.search_key,
            start=request.start,
            limit=request.limit,
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/message")
async def zoho_mail_message(request: ZohoMailMessageRequest) -> dict[str, object]:
    try:
        return await get_message_content(
            request.user_id,
            request.account_id,
            request.folder_id,
            request.message_id,
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/send")
async def zoho_mail_send(request: ZohoMailSendRequest) -> dict[str, object]:
    try:
        return await send_message(
            request.user_id,
            request.account_id,
            request.from_address,
            request.to_address,
            request.subject,
            request.content,
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)

@router.post("/reply")
async def zoho_mail_reply(request: ZohoMailReplyRequest) -> dict[str, object]:
    try:
        return await reply_to_message(
            request.user_id,
            request.account_id,
            request.message_id,
            request.from_address,
            request.to_address,
            request.subject,
            request.content,
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)



@router.post("/delete")
async def zoho_mail_delete(request: ZohoMailDeleteRequest) -> dict[str, object]:
    try:
        return await delete_message(
            request.user_id,
            request.account_id,
            request.folder_id,
            request.message_id,
            approved=request.approved,
            permanently=request.permanently,
        )
    except (PermissionError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/draft")
async def zoho_mail_save_draft(request: ZohoMailDraftRequest) -> dict[str, object]:
    try:
        return await save_draft(
            request.user_id,
            request.account_id,
            request.from_address,
            request.to_address,
            request.subject,
            request.content,
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)


@router.post("/draft/replace")
async def zoho_mail_replace_draft(request: ZohoMailReplaceDraftRequest) -> dict[str, object]:
    try:
        return await replace_draft(
            request.user_id,
            request.account_id,
            request.folder_id,
            request.draft_id,
            request.from_address,
            request.to_address,
            request.subject,
            request.content,
            approved=request.approved,
        )
    except (PermissionError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        _raise_provider_error(exc)
