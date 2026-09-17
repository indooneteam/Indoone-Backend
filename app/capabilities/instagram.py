from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token

_INTEGRATION_ID = "instagram"
_AUTH_URL = "https://www.instagram.com/oauth/authorize"
_TOKEN_URL = "https://api.instagram.com/oauth/access_token"
_GRAPH_URL = "https://graph.instagram.com"
_BASIC_SCOPE = "instagram_business_basic"
_PUBLISH_SCOPE = "instagram_business_content_publish"
_MANAGE_MESSAGES_SCOPE = "instagram_business_manage_messages"
_MANAGE_COMMENTS_SCOPE = "instagram_business_manage_comments"
_DEFAULT_SCOPES = f"{_BASIC_SCOPE},{_PUBLISH_SCOPE}"


def _credentials() -> tuple[str, str]:
    client_id = os.getenv("INDOONE_INSTAGRAM_APP_ID", "").strip()
    client_secret = os.getenv("INDOONE_INSTAGRAM_APP_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("instagram app credentials are not configured")
    return client_id, client_secret


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def build_instagram_authorization(
    state: str,
    redirect_uri: str,
    include_publishing: bool = True,
    include_messages: bool = False,
    include_comments: bool = False,
) -> dict[str, object]:
    client_id, _ = _credentials()
    scopes = [_BASIC_SCOPE]
    if include_publishing:
        scopes.append(_PUBLISH_SCOPE)
    if include_messages:
        scopes.append(_MANAGE_MESSAGES_SCOPE)
    if include_comments:
        scopes.append(_MANAGE_COMMENTS_SCOPE)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "scope": ",".join(scopes),
        "state": state.strip(),
        "force_reauth": "false",
    }
    return {
        "integration": _INTEGRATION_ID,
        "authorization_url": f"{_AUTH_URL}?{urlencode(params)}",
        "scope": ",".join(scopes),
        "state_required": True,
        "professional_accounts_only": True,
        "secrets_exposed": False,
    }


def _decode(row: dict[str, object], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored instagram token cannot be decrypted") from exc


def _expires_at(expires_in: object) -> str | None:
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    return None


def _store_token(user_id: str, body: dict[str, object]) -> dict[str, object]:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("instagram oauth response contains no access token")
    refresh_token = body.get("refresh_token")
    scope = body.get("permissions") or body.get("scope") or _DEFAULT_SCOPES
    cipher = _fernet()
    upsert_integration_token(
        user_id,
        _INTEGRATION_ID,
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if isinstance(refresh_token, str) and refresh_token else None,
        "Bearer",
        str(scope),
        _expires_at(body.get("expires_in")),
    )
    return {
        "scope": str(scope),
        "expires_at": _expires_at(body.get("expires_in")),
    }


async def exchange_instagram_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _INTEGRATION_ID)
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    client_id, client_secret = _credentials()
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "redirect_uri": state_data["redirect_uri"],
        "code": code.strip(),
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(_TOKEN_URL, data=payload, headers={"Accept": "application/json"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram oauth returned an invalid response")

    short_token = body.get("access_token")
    if not isinstance(short_token, str) or not short_token:
        data = body.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            short_token = data[0].get("access_token")
            body = {**data[0], **body}
    if not isinstance(short_token, str) or not short_token:
        raise RuntimeError("instagram oauth response contains no short-lived access token")

    long_lived_params = {
        "grant_type": "ig_exchange_token",
        "client_secret": client_secret,
        "access_token": short_token,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GRAPH_URL}/access_token", params=long_lived_params, headers={"Accept": "application/json"})
        response.raise_for_status()
        long_body = response.json()
    if not isinstance(long_body, dict):
        raise RuntimeError("instagram long-lived token response is invalid")
    stored = _store_token(state_data["user_id"], {**body, **long_body})
    return {
        "integration": _INTEGRATION_ID,
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": stored["scope"],
        "expires_at": stored["expires_at"],
        "professional_accounts_only": True,
        "secrets_exposed": False,
    }


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if row is None:
        raise ValueError("instagram is not connected for user")
    return row


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decode(row, "access_token")
    if not token:
        raise RuntimeError("stored instagram access token is empty")
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(days=7):
                refreshed = await _refresh_access_token(user_id, row)
                return refreshed
        except ValueError:
            pass
    return token


async def _refresh_access_token(user_id: str, row: dict[str, object]) -> str:
    token = _decode(row, "access_token")
    if not token:
        raise RuntimeError("stored instagram access token is empty")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GRAPH_URL}/refresh_access_token",
            params={"grant_type": "ig_refresh_token", "access_token": token},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram refresh returned an invalid response")
    _store_token(user_id, {**body, "access_token": body.get("access_token") or token})
    refreshed = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if refreshed is None:
        raise RuntimeError("refreshed instagram token could not be loaded")
    refreshed_token = _decode(refreshed, "access_token")
    if not refreshed_token:
        raise RuntimeError("refreshed instagram token is empty")
    return refreshed_token


async def get_profile(user_id: str) -> dict[str, object]:
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GRAPH_URL}/me",
            params={"fields": "id,username,name,account_type,profile_picture_url,followers_count,follows_count,media_count"},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram profile response is invalid")
    return {"integration": _INTEGRATION_ID, "profile": body, "secrets_exposed": False}


async def list_media(user_id: str, limit: int = 25, after: str = "") -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    token = await _access_token(user_id)
    params: dict[str, object] = {
        "fields": "id,caption,media_type,media_product_type,media_url,permalink,thumbnail_url,timestamp,username",
        "limit": limit,
    }
    if after:
        params["after"] = after
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GRAPH_URL}/me/media", params=params, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram media response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def create_media_container(
    user_id: str,
    image_url: str = "",
    video_url: str = "",
    caption: str = "",
    media_type: str = "IMAGE",
) -> dict[str, object]:
    media_type = media_type.strip().upper()
    if media_type not in {"IMAGE", "VIDEO", "REELS", "STORIES", "CAROUSEL"}:
        raise ValueError("media_type must be IMAGE, VIDEO, REELS, STORIES, or CAROUSEL")
    if media_type in {"IMAGE", "CAROUSEL", "STORIES"} and not image_url:
        raise ValueError("image_url is required for this media_type")
    if media_type in {"VIDEO", "REELS"} and not video_url:
        raise ValueError("video_url is required for this media_type")
    if not (image_url or video_url):
        raise ValueError("a media URL is required")
    token = await _access_token(user_id)
    data: dict[str, object] = {"caption": caption, "access_token": token}
    if media_type == "IMAGE":
        data["image_url"] = image_url
    elif media_type == "VIDEO":
        data.update({"media_type": "VIDEO", "video_url": video_url})
    else:
        data.update({"media_type": media_type, "video_url": video_url})
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{_GRAPH_URL}/me/media", data=data, headers={"Accept": "application/json"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict) or not body.get("id"):
        raise RuntimeError("instagram media container response is invalid")
    return {"integration": _INTEGRATION_ID, "container": body, "secrets_exposed": False}


async def publish_media(user_id: str, creation_id: str, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram publishing requires explicit approval")
    if not creation_id.strip():
        raise ValueError("creation_id is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_GRAPH_URL}/me/media_publish",
            data={"creation_id": creation_id.strip(), "access_token": token},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram publish response is invalid")
    return {"integration": _INTEGRATION_ID, "published": True, "result": body, "secrets_exposed": False}
