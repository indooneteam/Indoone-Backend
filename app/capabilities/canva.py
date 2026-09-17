from __future__ import annotations

import base64
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token

_INTEGRATION_ID = "canva"
_AUTH_URL = "https://www.canva.com/api/oauth/authorize"
_TOKEN_URL = "https://api.canva.com/rest/v1/oauth/token"
_API_URL = "https://api.canva.com/rest/v1"
_READ_SCOPE = "design:content:read"
_WRITE_SCOPE = "design:content:write"
_ASSET_SCOPE = "asset:read"


def _credentials() -> tuple[str, str]:
    client_id = os.getenv("INDOONE_CANVA_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_CANVA_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("canva client credentials are not configured")
    return client_id, client_secret


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def _pkce_verifier() -> str:
    return secrets.token_urlsafe(64)


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_canva_authorization(state: str, redirect_uri: str, include_write: bool = True) -> dict[str, object]:
    client_id, _ = _credentials()
    verifier = _pkce_verifier()
    scopes = [_READ_SCOPE]
    if include_write:
        scopes.append(_WRITE_SCOPE)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state.strip(),
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    return {
        "integration": _INTEGRATION_ID,
        "authorization_url": f"{_AUTH_URL}?{urlencode(params)}",
        "scope": " ".join(scopes),
        "pkce_code_verifier": verifier,
        "state_required": True,
        "secrets_exposed": False,
    }


def _decode(row: dict[str, object], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored canva token cannot be decrypted") from exc


def _expires_at(expires_in: object) -> str | None:
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    return None


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("canva oauth response contains no access token")
    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = fallback_refresh
    scope = str(body.get("scope") or "")
    cipher = _fernet()
    upsert_integration_token(
        user_id,
        _INTEGRATION_ID,
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        "Bearer",
        scope,
        _expires_at(body.get("expires_in")),
    )


async def exchange_canva_code(state: str, code: str, code_verifier: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _INTEGRATION_ID)
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    if not code_verifier.strip():
        raise ValueError("pkce code verifier is required")
    client_id, client_secret = _credentials()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    payload = {
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": state_data["redirect_uri"],
        "code_verifier": code_verifier.strip(),
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data=payload,
            headers={"Authorization": f"Basic {basic}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva oauth returned an invalid response")
    _store_token(state_data["user_id"], body)
    return {
        "integration": _INTEGRATION_ID,
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": str(body.get("scope") or ""),
        "expires_at": _expires_at(body.get("expires_in")),
        "secrets_exposed": False,
    }


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if row is None:
        raise ValueError("canva is not connected for user")
    return row


async def _refresh_access_token(user_id: str, row: dict[str, object]) -> str:
    refresh_token = _decode(row, "refresh_token")
    if not refresh_token:
        raise RuntimeError("canva access token expired and no refresh token is stored")
    client_id, client_secret = _credentials()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            headers={"Authorization": f"Basic {basic}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva refresh returned an invalid response")
    _store_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if refreshed is None:
        raise RuntimeError("refreshed canva token could not be loaded")
    token = _decode(refreshed, "access_token")
    if not token:
        raise RuntimeError("refreshed canva token is empty")
    return token


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decode(row, "access_token")
    if not token:
        raise RuntimeError("stored canva access token is empty")
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(seconds=30):
                return await _refresh_access_token(user_id, row)
        except ValueError:
            pass
    return token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def list_designs(user_id: str, query: str = "", continuation: str = "") -> dict[str, object]:
    token = await _access_token(user_id)
    params: dict[str, object] = {"query": query} if query.strip() else {}
    if continuation.strip():
        params["continuation"] = continuation.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_URL}/designs", params=params, headers=_headers(token))
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva designs response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def get_design(user_id: str, design_id: str) -> dict[str, object]:
    if not design_id.strip():
        raise ValueError("design_id is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_URL}/designs/{design_id.strip()}", headers=_headers(token))
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva design response is invalid")
    return {"integration": _INTEGRATION_ID, "design": body, "secrets_exposed": False}


async def get_export_formats(user_id: str, design_id: str) -> dict[str, object]:
    if not design_id.strip():
        raise ValueError("design_id is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_URL}/designs/{design_id.strip()}/export-formats", headers=_headers(token))
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva export formats response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def create_export_job(user_id: str, design_id: str, format_payload: dict[str, object], approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("canva export requires explicit approval")
    if not design_id.strip():
        raise ValueError("design_id is required")
    if not isinstance(format_payload, dict) or not format_payload:
        raise ValueError("format is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_API_URL}/exports",
            json={"design_id": design_id.strip(), "format": format_payload},
            headers={**_headers(token), "Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva export job response is invalid")
    return {"integration": _INTEGRATION_ID, "created": True, "export": body, "secrets_exposed": False}


async def get_export_job(user_id: str, export_id: str) -> dict[str, object]:
    if not export_id.strip():
        raise ValueError("export_id is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_URL}/exports/{export_id.strip()}", headers=_headers(token))
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("canva export job response is invalid")
    return {"integration": _INTEGRATION_ID, "export": body, "secrets_exposed": False}
