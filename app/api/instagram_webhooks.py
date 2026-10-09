from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from app.capabilities.channel_ai_reply import process_instagram_webhook
from app.capabilities.instagram_webhooks import normalize_event, verify_challenge, verify_signature
from app.capabilities.control_center import intake_enabled

router = APIRouter(prefix="/integrations/instagram/webhook", tags=["instagram-webhooks"])


@router.get("")
async def verify(
    hub_mode: str = "",
    hub_challenge: str = "",
    hub_verify_token: str = "",
) -> Response:
    try:
        challenge = verify_challenge(hub_mode, hub_challenge, hub_verify_token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=challenge, media_type="text/plain")


@router.post("")
async def receive(request: Request, background_tasks: BackgroundTasks) -> dict[str, object]:
    signature = request.headers.get("X-Hub-Signature-256", "")
    raw_body = await request.body()
    try:
        verify_signature(raw_body, signature)
        if not intake_enabled("instagram"):
            request.state.control_intake_blocked = True
            return {"integration": "instagram", "received": True, "processed": False, "reason": "intake_paused"}
        payload = json.loads(raw_body.decode("utf-8"))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="instagram webhook payload is invalid JSON") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="instagram webhook payload must be an object")

    try:
        normalized = normalize_event(payload)
        background_tasks.add_task(process_instagram_webhook, payload)
        return normalized
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
