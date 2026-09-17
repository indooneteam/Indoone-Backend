from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.capabilities.whatsapp import parse_webhook, probe_whatsapp, send_text_message, validate_signature, verify_webhook

router = APIRouter(prefix="/integrations/whatsapp", tags=["whatsapp"])


class SendTextRequest(BaseModel):
    to: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4096)
    approved: bool = False
    preview_url: bool = False


@router.get("/probe")
async def probe() -> dict[str, object]:
    try:
        return await probe_whatsapp()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/send-text")
async def send_text(request: SendTextRequest) -> dict[str, object]:
    try:
        return await send_text_message(request.to, request.text, request.approved, request.preview_url)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/webhook")
async def verify(request: Request) -> str:
    try:
        return verify_webhook(request.query_params.get("hub.mode"), request.query_params.get("hub.verify_token"), request.query_params.get("hub.challenge"))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/webhook")
async def webhook(request: Request) -> dict[str, object]:
    raw = await request.body()
    app_secret = os.getenv("INDOONE_WHATSAPP_APP_SECRET", "").strip()
    if not app_secret:
        raise HTTPException(status_code=503, detail="whatsapp webhook signature validation is not configured")
    if not validate_signature(app_secret, request.headers.get("X-Hub-Signature-256"), raw):
        raise HTTPException(status_code=403, detail="invalid whatsapp webhook signature")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid webhook JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="webhook payload must be an object")
    return {"integration": "whatsapp_business", "messages": parse_webhook(payload), "received": True}
