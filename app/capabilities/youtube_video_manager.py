from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token
from app.capabilities.youtube import _auth_headers, _fernet, get_videos

_API_BASE = "https://www.googleapis.com/youtube/v3"
_UPLOAD_THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
_OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_YOUTUBE_MANAGER_SCOPE = "https://www.googleapis.com/auth/youtube"
_MANAGER_STATE_INTEGRATION = "youtube_video_manager"
_MAX_THUMBNAIL_BYTES = 50 * 1024 * 1024
_ALLOWED_THUMBNAIL_MIME_TYPES = {"image/jpeg", "image/png"}


def build_youtube_manager_authorization(state: str, redirect_uri: str) -> dict[str, object]:
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    if not client_id or not os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip():
        raise RuntimeError("google oauth client credentials are not configured")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "scope": _YOUTUBE_MANAGER_SCOPE,
        "state": state.strip(),
    }
    return {
        "integration": "youtube",
        "connect_mode": "channel_manager",
        "authorization_url": f"{_OAUTH_URL}?{urlencode(params)}",
        "scope": _YOUTUBE_MANAGER_SCOPE,
        "capabilities": [
            "edit video title, description, category, tags, and privacy",
            "delete videos with explicit approval",
            "set custom video thumbnails with explicit approval",
        ],
        "secrets_exposed": False,
    }


def _store_manager_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("youtube manager oauth token response contains no access token")
    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = fallback_refresh
    expires_at: str | None = None
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    cipher = _fernet()
    scope = str(body.get("scope") or _YOUTUBE_MANAGER_SCOPE)
    upsert_integration_token(
        user_id,
        "youtube",
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        str(body.get("token_type") or "Bearer"),
        scope,
        expires_at,
    )


async def exchange_youtube_manager_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _MANAGER_STATE_INTEGRATION)
    if state_data is None:
        raise ValueError("invalid or expired youtube video manager oauth state")
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code.strip(),
                "redirect_uri": state_data["redirect_uri"],
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube manager oauth returned an invalid response")
    _store_manager_token(state_data["user_id"], body)
    return {
        "integration": "youtube",
        "user_id": state_data["user_id"],
        "connected": True,
        "connect_mode": "channel_manager",
        "scope": str(body.get("scope") or _YOUTUBE_MANAGER_SCOPE),
        "secrets_exposed": False,
    }


def _manager_scope(user_id: str) -> str:
    row = get_integration_token(user_id.strip(), "youtube")
    if row is None:
        raise ValueError("integration is not connected for user")
    scope = str(row.get("scope") or "")
    if _YOUTUBE_MANAGER_SCOPE not in scope:
        raise PermissionError("youtube video manager access is not connected; authorize channel manager access")
    return scope


async def _manager_refresh(user_id: str, row: dict[str, object]) -> str:
    refresh_raw = row.get("refresh_token")
    if refresh_raw is None:
        raise RuntimeError("youtube manager access token expired and no refresh token is stored")
    try:
        refresh_token = _fernet().decrypt(bytes(refresh_raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored youtube manager refresh token cannot be decrypted") from exc
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube manager oauth refresh returned an invalid response")
    _store_manager_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = get_integration_token(user_id.strip(), "youtube")
    if refreshed is None:
        raise RuntimeError("refreshed youtube manager token could not be loaded")
    raw = refreshed.get("access_token")
    if raw is None:
        raise RuntimeError("refreshed youtube manager access token is empty")
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("refreshed youtube manager access token cannot be decrypted") from exc


async def _manager_token(user_id: str) -> str:
    _manager_scope(user_id)
    row = get_integration_token(user_id.strip(), "youtube")
    if row is None:
        raise ValueError("integration is not connected for user")
    raw = row.get("access_token")
    if raw is None:
        raise RuntimeError("stored youtube manager access token is empty")
    try:
        token = _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored youtube manager access token cannot be decrypted") from exc
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(seconds=30):
                return await _manager_refresh(user_id, row)
        except ValueError:
            pass
    return token


def build_video_update_payload(
    current_video: dict[str, object],
    *,
    title: str | None = None,
    description: str | None = None,
    category_id: str | None = None,
    privacy_status: str | None = None,
    tags: list[str] | None = None,
) -> tuple[list[str], dict[str, object]]:
    video_id = str(current_video.get("id") or "").strip()
    if not video_id:
        raise ValueError("video_id is required")
    if all(value is None for value in (title, description, category_id, privacy_status, tags)):
        raise ValueError("at least one video field must be provided")

    snippet = current_video.get("snippet") if isinstance(current_video.get("snippet"), dict) else {}
    snippet_update_requested = any(value is not None for value in (title, description, category_id, tags))
    status_update_requested = privacy_status is not None

    resource: dict[str, object] = {"id": video_id}
    parts: list[str] = []

    if snippet_update_requested:
        resolved_title = str(title if title is not None else snippet.get("title") or "").strip()
        resolved_category = str(category_id if category_id is not None else snippet.get("categoryId") or "").strip()
        if not resolved_title:
            raise ValueError("title is required for snippet updates")
        if not resolved_category:
            raise ValueError("category_id is required for snippet updates")
        snippet_resource: dict[str, object] = {
            "title": resolved_title,
            "categoryId": resolved_category,
            "description": str(description if description is not None else snippet.get("description") or ""),
        }
        if tags is not None:
            snippet_resource["tags"] = [item.strip() for item in tags if item.strip()]
        elif isinstance(snippet.get("tags"), list):
            snippet_resource["tags"] = [str(item) for item in snippet["tags"]]
        resource["snippet"] = snippet_resource
        parts.append("snippet")

    if status_update_requested:
        normalized_privacy = str(privacy_status).strip().lower()
        if normalized_privacy not in {"private", "unlisted", "public"}:
            raise ValueError("privacy_status must be private, unlisted, or public")
        resource["status"] = {"privacyStatus": normalized_privacy}
        parts.append("status")

    return parts, resource


async def update_video(
    user_id: str,
    video_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    category_id: str | None = None,
    privacy_status: str | None = None,
    tags: list[str] | None = None,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube video update operations")
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id is required")
    existing = await get_videos([video_id], user_id)
    videos = existing.get("videos") if isinstance(existing.get("videos"), list) else []
    if not videos:
        raise ValueError("video not found")
    parts, payload = build_video_update_payload(
        videos[0],
        title=title,
        description=description,
        category_id=category_id,
        privacy_status=privacy_status,
        tags=tags,
    )
    token = await _manager_token(user_id)
    params = {"part": ",".join(parts)}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(
            f"{_API_BASE}/videos",
            params=params,
            headers=_auth_headers(token),
            json=payload,
        )
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            response = await client.put(
                f"{_API_BASE}/videos",
                params=params,
                headers=_auth_headers(token),
                json=payload,
            )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("youtube returned an invalid video update response")
    return {
        "integration": "youtube",
        "operation": "update_video",
        "video": {
            "id": result.get("id"),
            "snippet": result.get("snippet"),
            "status": result.get("status"),
        },
        "secrets_exposed": False,
    }


async def delete_video(user_id: str, video_id: str, *, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube video delete operations")
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id is required")
    token = await _manager_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(
            f"{_API_BASE}/videos",
            params={"id": video_id},
            headers=_auth_headers(token),
        )
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            response = await client.delete(
                f"{_API_BASE}/videos",
                params={"id": video_id},
                headers=_auth_headers(token),
            )
        response.raise_for_status()
    return {
        "integration": "youtube",
        "operation": "delete_video",
        "video_id": video_id,
        "deleted": True,
        "secrets_exposed": False,
    }


async def set_video_thumbnail(
    user_id: str,
    video_id: str,
    content: bytes,
    mime_type: str,
    *,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube thumbnail operations")
    video_id = video_id.strip()
    mime_type = mime_type.strip().lower()
    if not video_id:
        raise ValueError("video_id is required")
    if not content:
        raise ValueError("thumbnail content is required")
    if len(content) > _MAX_THUMBNAIL_BYTES:
        raise ValueError("thumbnail exceeds the 50MB YouTube limit")
    if mime_type not in _ALLOWED_THUMBNAIL_MIME_TYPES:
        raise ValueError("thumbnail mime_type must be image/jpeg or image/png")
    token = await _manager_token(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": mime_type,
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _UPLOAD_THUMBNAIL_URL,
            params={"videoId": video_id},
            headers=headers,
            content=content,
        )
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.post(
                _UPLOAD_THUMBNAIL_URL,
                params={"videoId": video_id},
                headers=headers,
                content=content,
            )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("youtube returned an invalid thumbnail response")
    return {
        "integration": "youtube",
        "operation": "set_video_thumbnail",
        "video_id": video_id,
        "thumbnails": result.get("items") if isinstance(result.get("items"), list) else [],
        "secrets_exposed": False,
    }
