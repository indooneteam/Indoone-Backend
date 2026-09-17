from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_messaging import (
    get_message,
    list_conversation_messages,
    list_conversations,
    send_media_share,
    send_text_message,
)

router = APIRouter(prefix="/integrations/instagram/messaging", tags=["instagram-messaging"])


class ConversationsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=25, ge=1, le=100)
    after: str = Field(default="", max_length=2048)
    target_user_id: str = Field(default="", max_length=256)


class ConversationMessagesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)


class MessageRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)


class SendTextRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    recipient_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1000)
    approved: bool = False


class SendMediaShareRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    recipient_id: str = Field(min_length=1, max_length=256)
    media_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


def _provider_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=502, detail=f"instagram messaging provider failed: {exc}")


@router.post("/conversations")
async def conversations(request: ConversationsRequest) -> dict[str, object]:
    try:
        return await list_conversations(
            request.user_id,
            request.limit,
            request.after,
            request.target_user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise _provider_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/conversation/messages")
async def conversation_messages(request: ConversationMessagesRequest) -> dict[str, object]:
    try:
        return await list_conversation_messages(request.user_id, request.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise _provider_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/message")
async def message(request: MessageRequest) -> dict[str, object]:
    try:
        return await get_message(request.user_id, request.message_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise _provider_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/send")
async def send(request: SendTextRequest) -> dict[str, object]:
    try:
        return await send_text_message(
            request.user_id,
            request.recipient_id,
            request.text,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise _provider_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/send-media")
async def send_media(request: SendMediaShareRequest) -> dict[str, object]:
    try:
        return await send_media_share(
            request.user_id,
            request.recipient_id,
            request.media_id,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise _provider_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
