from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token

_INTEGRATION_ID = "facebook"
_GRAPH_URL = "https://graph.facebook.com"
_AUTH_URL = "https://www.facebook.com/dialog/oauth"
_TOKEN_URL = f"{_GRAPH_URL}/oauth/access_token"
_SCOPES = "pages_show_list,pages_read_engagement,pages_manage_posts"


def _credentials() -> tuple[str, str]:
    app_id = os.getenv("INDOONE_FACEBOOK_APP_ID", "").strip()
    app_secret = os.getenv("INDOONE_FACEBOOK_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise RuntimeError("facebook app credentials are not configured")
    return app_id, app_secret


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def build_facebook_authorization(state: str, redirect_uri: str) -> dict[str, object]:
    app_id, _ = _credentials()
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "scope": _SCOPES,
        "state": state.strip(),
    }
    return {
        "integration": _INTEGRATION_ID,
        "authorization_url": f"{_AUTH_URL}?{urlencode(params)}",
        "scope": _SCOPES,
        "state_required": True,
        "pages_only": True,
        "secrets_exposed": False,
    }


def _decode(row: dict[str, object], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored facebook token cannot be decrypted") from exc


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    token = body.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("facebook oauth response contains no access token")
    expires_at = None
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    refresh = body.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        refresh = fallback_refresh
    cipher = _fernet()
    upsert_integration_token(
        user_id,
        _INTEGRATION_ID,
        cipher.encrypt(token.encode("utf-8")),
        cipher.encrypt(refresh.encode("utf-8")) if refresh else None,
        "Bearer",
        str(body.get("scope") or _SCOPES),
        expires_at,
    )


async def exchange_facebook_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _INTEGRATION_ID)
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    app_id, app_secret = _credentials()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            _TOKEN_URL,
            params={
                "client_id": app_id,
                "client_secret": app_secret,
                "redirect_uri": state_data["redirect_uri"],
                "code": code.strip(),
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("facebook oauth returned an invalid response")
    _store_token(state_data["user_id"], body)
    return {
        "integration": _INTEGRATION_ID,
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": str(body.get("scope") or _SCOPES),
        "expires_at": _expires_at_from_response(body),
        "pages_only": True,
        "secrets_exposed": False,
    }


def _expires_at_from_response(body: dict[str, object]) -> str | None:
    value = body.get("expires_in")
    if isinstance(value, (int, float)) and value > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=int(value))).isoformat()
    return None


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if row is None:
        raise ValueError("facebook is not connected for user")
    return row


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decode(row, "access_token")
    if not token:
        raise RuntimeError("stored facebook access token is empty")
    return token


async def list_pages(user_id: str) -> dict[str, object]:
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GRAPH_URL}/me/accounts",
            params={"fields": "id,name,category,tasks"},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("facebook pages response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def page_access_token(user_id: str, page_id: str) -> str:
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GRAPH_URL}/me/accounts",
            params={"fields": "id,access_token", "limit": 100},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    entries = body.get("data") if isinstance(body, dict) else None
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and str(entry.get("id")) == page_id.strip():
                page_token = entry.get("access_token")
                if isinstance(page_token, str) and page_token:
                    return page_token
    raise ValueError("facebook page is not available for this connected user")


async def list_page_posts(user_id: str, page_id: str, limit: int = 25, after: str = "") -> dict[str, object]:
    if not page_id.strip():
        raise ValueError("page_id is required")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    page_token = await page_access_token(user_id, page_id)
    params: dict[str, object] = {"fields": "id,message,created_time,permalink_url,attachments", "limit": limit}
    if after:
        params["after"] = after
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GRAPH_URL}/{page_id.strip()}/posts", params=params, headers={"Authorization": f"Bearer {page_token}", "Accept": "application/json"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("facebook posts response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def create_page_post(user_id: str, page_id: str, message: str, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("facebook page publishing requires explicit approval")
    if not page_id.strip() or not message.strip():
        raise ValueError("page_id and message are required")
    token = await page_access_token(user_id, page_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_GRAPH_URL}/{page_id.strip()}/feed",
            data={"message": message.strip(), "access_token": token},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("facebook publish response is invalid")
    return {"integration": _INTEGRATION_ID, "published": True, "result": body, "secrets_exposed": False}
