from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.telegram import (
    delete_webhook,
    get_updates,
    get_webhook_info,
    probe_telegram,
    send_message,
    set_webhook,
    validate_webhook_secret,
)

router = APIRouter(prefix="/telegram", tags=["telegram"])


class TelegramSendRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=4096)
    approved: bool = False
    parse_mode: str = Field(default="", max_length=16)


class TelegramUpdatesRequest(BaseModel):
    offset: int | None = Field(default=None, ge=-2_147_483_648, le=2_147_483_647)
    limit: int = Field(default=100, ge=1, le=100)
    timeout: int = Field(default=0, ge=0, le=50)


class TelegramWebhookRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    approved: bool = False
    drop_pending_updates: bool = False


class TelegramWebhookDeleteRequest(BaseModel):
    approved: bool = False
    drop_pending_updates: bool = False


@router.get("/probe")
async def telegram_probe() -> dict[str, object]:
    try:
        return await probe_telegram()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/updates")
async def telegram_updates(request: TelegramUpdatesRequest) -> dict[str, object]:
    try:
        return await get_updates(request.offset, request.limit, request.timeout)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/send-message")
async def telegram_send_message(request: TelegramSendRequest) -> dict[str, object]:
    try:
        return await send_message(request.chat_id, request.text, request.approved, request.parse_mode)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/webhook")
async def telegram_webhook_info() -> dict[str, object]:
    try:
        return await get_webhook_info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/webhook")
async def telegram_set_webhook(request: TelegramWebhookRequest) -> dict[str, object]:
    try:
        return await set_webhook(request.url, request.approved, request.drop_pending_updates)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/webhook")
async def telegram_delete_webhook(request: TelegramWebhookDeleteRequest) -> dict[str, object]:
    try:
        return await delete_webhook(request.approved, request.drop_pending_updates)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/webhook/incoming")
async def telegram_incoming(update: dict[str, object], x_telegram_bot_api_secret_token: str | None = Header(default=None)) -> dict[str, bool]:
    if not validate_webhook_secret(x_telegram_bot_api_secret_token):
        raise HTTPException(status_code=401, detail="invalid telegram webhook secret")
    return {"accepted": True}
