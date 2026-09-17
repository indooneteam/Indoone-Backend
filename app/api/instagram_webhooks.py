from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, Response

from app.capabilities.instagram_webhooks import normalize_event, verify_challenge, verify_signature

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
async def receive(request: Request) -> dict[str, object]:
    signature = request.headers.get("X-Hub-Signature-256", "")
    raw_body = await request.body()
    try:
        verify_signature(raw_body, signature)
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
        return normalize_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
