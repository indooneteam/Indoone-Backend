from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.gmail.service import (
    create_gmail_draft,
    delete_gmail_draft,
    delete_gmail_message,
    get_gmail_message,
    list_gmail_drafts,
    list_gmail_messages,
    reply_gmail_message,
    send_gmail_message,
    update_gmail_draft,
)


router = APIRouter(prefix="/gmail", tags=["gmail"])


class GmailMessagesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    query: str = Field(default="", max_length=500)
    page_token: str = Field(default="", max_length=2048)
    max_results: int = Field(default=20, ge=1, le=100)


class GmailMessageRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)


class GmailSendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    to: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=2000)
    body: str = Field(min_length=1, max_length=20000)
    approved: bool = False


class GmailDraftRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    to: str = Field(default="", max_length=512)
    subject: str = Field(default="", max_length=2000)
    body: str = Field(default="", max_length=20000)


class GmailDraftActionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    draft_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


class GmailDraftUpdateRequest(GmailDraftRequest):
    draft_id: str = Field(min_length=1, max_length=256)


class GmailMessageActionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


class GmailReplyRequest(GmailMessageActionRequest):
    body: str = Field(min_length=1, max_length=20000)


@router.post("/messages")
async def gmail_messages(request: GmailMessagesRequest) -> dict[str, object]:
    try:
        return await list_gmail_messages(request.user_id, request.query, request.page_token, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/message")
async def gmail_message(request: GmailMessageRequest) -> dict[str, object]:
    try:
        return await get_gmail_message(request.user_id, request.message_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/send")
async def gmail_send(request: GmailSendRequest) -> dict[str, object]:
    try:
        return await send_gmail_message(request.user_id, request.to, request.subject, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc



@router.post("/drafts/list")
async def gmail_drafts_list(request: GmailMessagesRequest) -> dict[str, object]:
    try:
        return await list_gmail_drafts(request.user_id, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to list drafts")


@router.post("/drafts")
async def gmail_create_draft(request: GmailDraftRequest) -> dict[str, object]:
    try:
        return await create_gmail_draft(request.user_id, request.to, request.subject, request.body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to create draft")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("/drafts")
async def gmail_update_draft(request: GmailDraftUpdateRequest) -> dict[str, object]:
    try:
        return await update_gmail_draft(
            request.user_id,
            request.draft_id,
            request.to,
            request.subject,
            request.body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to update draft")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/drafts/delete")
async def gmail_delete_draft(request: GmailDraftActionRequest) -> dict[str, object]:
    try:
        return await delete_gmail_draft(request.user_id, request.draft_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to delete draft")


@router.post("/messages/delete")
async def gmail_delete_message(request: GmailMessageActionRequest) -> dict[str, object]:
    try:
        return await delete_gmail_message(request.user_id, request.message_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to move message to Trash")


@router.post("/reply")
async def gmail_reply(request: GmailReplyRequest) -> dict[str, object]:
    try:
        return await reply_gmail_message(
            request.user_id,
            request.message_id,
            request.body,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Gmail provider failed to send reply")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
