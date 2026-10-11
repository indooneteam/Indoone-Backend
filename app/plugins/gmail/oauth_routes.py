from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.integrations import exchange_gmail_server_auth_code, gmail_server_authorization_config


router = APIRouter(tags=["gmail"])


class GmailServerAuthCodeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    server_auth_code: str = Field(min_length=1, max_length=8_000)
    scope: str = Field(default="", max_length=4_000)


@router.get("/integrations/gmail/auth-config")
async def gmail_auth_config() -> dict[str, object]:
    try:
        # Only the public client ID is returned; secrets remain server-side.
        return gmail_server_authorization_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/integrations/gmail/server-auth-code")
async def gmail_server_auth_code(request: GmailServerAuthCodeRequest) -> dict[str, object]:
    try:
        return await exchange_gmail_server_auth_code(
            request.user_id,
            request.server_auth_code,
            request.scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Google rejected the Gmail authorization code") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
