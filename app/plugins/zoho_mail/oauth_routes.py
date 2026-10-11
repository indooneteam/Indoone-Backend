"""Zoho Mail OAuth endpoints. Tokens never leave the backend."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.capabilities.store import get_integration_token_metadata
from app.plugins.zoho_mail.oauth import (
    cancel_zoho_mail_authorization,
    exchange_zoho_mail_authorization_code,
    start_zoho_mail_authorization,
    zoho_mail_auth_config,
)

router = APIRouter(tags=["zoho-mail-oauth"])


class ZohoMailStartRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


@router.get("/integrations/zoho-mail/auth-config")
async def zoho_mail_auth_configuration() -> dict[str, object]:
    try:
        return zoho_mail_auth_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/integrations/zoho-mail/start")
async def zoho_mail_start(request: ZohoMailStartRequest) -> dict[str, object]:
    try:
        return start_zoho_mail_authorization(request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/integrations/zoho-mail/callback", response_class=HTMLResponse)
async def zoho_mail_callback(request: Request) -> HTMLResponse:
    # Read provider callback fields explicitly so OAuth denial callbacks can omit
    # the authorization code without being rejected during request validation.
    params = request.query_params
    code = params.get("code", "").strip()
    state = params.get("state", "").strip()
    error = params.get("error", "").strip()

    if len(code) > 8_000 or len(state) > 512 or len(error) > 256:
        raise HTTPException(status_code=400, detail="Zoho Mail callback parameters exceed allowed limits.")

    if error:
        cancel_zoho_mail_authorization(state)
        return HTMLResponse(
            "<!doctype html><html><body><h2>Zoho Mail connection was not completed</h2>"
            "<p>You can return to Indoone and try again.</p></body></html>",
            status_code=400,
        )

    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Zoho Mail callback is missing an authorization code or state.",
        )

    try:
        await exchange_zoho_mail_authorization_code(code, state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Zoho rejected the Mail authorization request") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return HTMLResponse(
        "<!doctype html><html><body><h2>Zoho Mail connected to Indoone</h2>"
        "<p>Authorization finished successfully. Return to Indoone and check the connection status.</p>"
        "</body></html>",
        status_code=200,
    )


@router.get("/integrations/zoho-mail/status")
@router.get("/integrations/zoho-mail/connection")
async def zoho_mail_connection_status(
    user_id: str = Query(..., min_length=1, max_length=256),
) -> dict[str, object]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    metadata = get_integration_token_metadata(normalized_user_id, "zoho_mail")
    if metadata is None:
        return {
            "integration": "zoho_mail",
            "user_id": normalized_user_id,
            "connected": False,
            "scope": "",
            "updated_at": None,
            "secrets_exposed": False,
        }
    return {
        "integration": "zoho_mail",
        "user_id": normalized_user_id,
        "connected": True,
        "scope": str(metadata.get("scope") or ""),
        "updated_at": metadata.get("updated_at"),
        "secrets_exposed": False,
    }
