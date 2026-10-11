from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.gmail.service import get_gmail_message, list_gmail_messages, send_gmail_message


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
