from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.capabilities.integrations import _fernet
from app.capabilities.store import delete_integration_token, get_integration_token, get_integration_token_metadata
from app.plugins.gmail.oauth import exchange_gmail_server_auth_code, gmail_server_authorization_config, store_native_access_token


router = APIRouter(tags=["gmail"])


class GmailServerAuthCodeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    server_auth_code: str = Field(min_length=1, max_length=8_000)
    scope: str = Field(default="", max_length=4_000)




class NativeIntegrationTokenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    access_token: str = Field(min_length=1, max_length=8_000)
    scope: str = Field(default="", max_length=4_000)
    expires_at: str | None = Field(default=None, max_length=128)


class GmailDisconnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)

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

@router.post("/integrations/gmail/disconnect")
async def gmail_disconnect(request: GmailDisconnectRequest) -> dict[str, object]:
    """Revoke provider access when possible, then delete only this user's local token."""
    user_id = request.user_id.strip()
    row = get_integration_token(user_id, "gmail")
    remote_revoked = False
    if row is not None:
        try:
            cipher = _fernet()
            encrypted_refresh = row.get("refresh_token")
            token = (
                cipher.decrypt(bytes(encrypted_refresh)).decode("utf-8")
                if encrypted_refresh
                else cipher.decrypt(bytes(row["access_token"])).decode("utf-8")
            )
        except Exception:
            token = ""

        if token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        "https://oauth2.googleapis.com/revoke",
                        data={"token": token},
                        headers={"Accept": "application/json"},
                    )
                    response.raise_for_status()
                    remote_revoked = True
            except httpx.HTTPError:
                # Remove local credentials even if provider revocation is temporarily
                # unavailable; never log the sensitive token or provider response body.
                remote_revoked = False

        delete_integration_token(user_id, "gmail")

    return {
        "integration": "gmail",
        "user_id": user_id,
        "connected": False,
        "remote_revoked": remote_revoked or row is None,
        "secrets_exposed": False,
    }


@router.get("/integrations/gmail/status")
async def gmail_connection_status(
    user_id: str = Query(..., min_length=1, max_length=256),
) -> dict[str, object]:
    """Return user-scoped Gmail connection metadata without exposing credentials."""
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    token_metadata = get_integration_token_metadata(normalized_user_id, "gmail")
    if token_metadata is None:
        return {
            "integration": "gmail",
            "user_id": normalized_user_id,
            "connected": False,
            "metadata": {},
            "secrets_exposed": False,
        }

    return {
        "integration": "gmail",
        "user_id": normalized_user_id,
        "connected": True,
        "metadata": {
            "scope": str(token_metadata.get("scope") or ""),
            "expires_at": token_metadata.get("expires_at"),
            "updated_at": token_metadata.get("updated_at"),
        },
        "secrets_exposed": False,
    }


@router.post("/integrations/gmail/native-token")
async def gmail_native_token(request: NativeIntegrationTokenRequest) -> dict[str, object]:
    try:
        return await store_native_access_token(
            request.user_id,
            "gmail",
            request.access_token,
            request.scope,
            request.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Google rejected the Gmail access token") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
