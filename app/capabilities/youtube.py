from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token

_API_BASE = "https://www.googleapis.com/youtube/v3"
_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
_OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
_YOUTUBE_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
_MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def _google_credentials() -> tuple[str, str]:
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    return client_id, client_secret


def build_youtube_authorization(state: str, redirect_uri: str, read_only: bool = False) -> dict[str, object]:
    client_id, _ = _google_credentials()
    scope = _YOUTUBE_READ_SCOPE if read_only else _YOUTUBE_SCOPE
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "scope": scope,
        "state": state.strip(),
    }
    return {
        "integration": "youtube",
        "authorization_url": f"{_OAUTH_URL}?{urlencode(params)}",
        "scope": scope,
        "state_required": True,
        "secrets_exposed": False,
    }


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("youtube oauth token response contains no access token")
    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = fallback_refresh
    expires_at: str | None = None
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    cipher = _fernet()
    upsert_integration_token(
        user_id,
        "youtube",
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        str(body.get("token_type") or "Bearer"),
        str(body.get("scope") or ""),
        expires_at,
    )


async def exchange_youtube_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), "youtube")
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    client_id, client_secret = _google_credentials()
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
        raise RuntimeError("youtube oauth returned an invalid response")
    _store_token(state_data["user_id"], body)
    return {
        "integration": "youtube",
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": str(body.get("scope") or ""),
        "secrets_exposed": False,
    }


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), "youtube")
    if row is None:
        raise ValueError("integration is not connected for user")
    return row


def _decrypt(row: dict[str, object], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored youtube token cannot be decrypted") from exc


async def _refresh(user_id: str, row: dict[str, object]) -> str:
    refresh_token = _decrypt(row, "refresh_token")
    if not refresh_token:
        raise RuntimeError("youtube access token expired and no refresh token is stored")
    client_id, client_secret = _google_credentials()
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
        raise RuntimeError("youtube oauth refresh returned an invalid response")
    _store_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = get_integration_token(user_id.strip(), "youtube")
    if refreshed is None:
        raise RuntimeError("refreshed youtube token could not be loaded")
    token = _decrypt(refreshed, "access_token")
    if not token:
        raise RuntimeError("refreshed youtube token is empty")
    return token


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decrypt(row, "access_token")
    if not token:
        raise RuntimeError("stored youtube access token is empty")
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(seconds=30):
                return await _refresh(user_id, row)
        except ValueError:
            pass
    return token


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _api_key() -> str:
    key = os.getenv("INDOONE_YOUTUBE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("youtube api key is not configured")
    return key


async def search_youtube(query: str, resource_type: str = "video", max_results: int = 10, page_token: str = "") -> dict[str, object]:
    query = query.strip()
    resource_type = resource_type.strip().lower()
    if not query:
        raise ValueError("query is required")
    if resource_type not in {"video", "channel", "playlist"}:
        raise ValueError("resource_type must be video, channel, or playlist")
    if not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50")
    params: dict[str, object] = {
        "part": "snippet",
        "q": query,
        "type": resource_type,
        "maxResults": max_results,
        "key": _api_key(),
    }
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_BASE}/search", params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid search response")
    return {
        "integration": "youtube",
        "query": query,
        "resource_type": resource_type,
        "items": body.get("items") if isinstance(body.get("items"), list) else [],
        "next_page_token": body.get("nextPageToken"),
        "total_results": body.get("pageInfo", {}).get("totalResults", 0) if isinstance(body.get("pageInfo"), dict) else 0,
        "secrets_exposed": False,
    }


async def get_videos(video_ids: list[str], user_id: str = "") -> dict[str, object]:
    ids = [item.strip() for item in video_ids if item.strip()]
    if not ids or len(ids) > 50:
        raise ValueError("video_ids must contain between 1 and 50 ids")
    params: dict[str, object] = {
        "part": "snippet,contentDetails,statistics,status",
        "id": ",".join(ids),
    }
    headers: dict[str, str] | None = None
    if user_id.strip():
        token = await _access_token(user_id)
        headers = _auth_headers(token)
    else:
        params["key"] = _api_key()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_BASE}/videos", params=params, headers=headers)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid videos response")
    return {
        "integration": "youtube",
        "videos": body.get("items") if isinstance(body.get("items"), list) else [],
        "secrets_exposed": False,
    }


async def get_my_channel(user_id: str) -> dict[str, object]:
    token = await _access_token(user_id)
    params = {"part": "snippet,contentDetails,statistics", "mine": "true"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_API_BASE}/channels", headers=_auth_headers(token), params=params)
        if response.status_code == 401:
            token = await _refresh(user_id, _token_row(user_id))
            response = await client.get(f"{_API_BASE}/channels", headers=_auth_headers(token), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid channel response")
    return {"integration": "youtube", "channels": body.get("items") if isinstance(body.get("items"), list) else [], "secrets_exposed": False}


async def upload_video(
    user_id: str,
    content: bytes,
    title: str,
    description: str = "",
    privacy_status: str = "private",
    category_id: str = "22",
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube upload operations")
    if not content:
        raise ValueError("video content is required")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise ValueError("video exceeds the supported API gateway size limit")
    title = title.strip()
    if not title:
        raise ValueError("title is required")
    privacy_status = privacy_status.strip().lower()
    if privacy_status not in {"private", "unlisted", "public"}:
        raise ValueError("privacy_status must be private, unlisted, or public")
    token = await _access_token(user_id)
    metadata = {
        "snippet": {"title": title, "description": description, "categoryId": category_id.strip() or "22"},
        "status": {"privacyStatus": privacy_status},
    }
    separator = b"\r\n"
    boundary = b"----IndooneYouTubeBoundary"
    body = (
        b"--" + boundary + separator
        + b'Content-Type: application/json; charset=UTF-8' + separator + separator
        + __import__("json").dumps(metadata).encode("utf-8") + separator
        + b"--" + boundary + separator
        + b"Content-Type: video/mp4" + separator + separator
        + content + separator
        + b"--" + boundary + b"--" + separator
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/related; boundary={boundary.decode()}",
        "Accept": "application/json",
    }
    params = {"part": "snippet,status"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(_UPLOAD_URL, params=params, headers=headers, content=body)
        if response.status_code == 401:
            token = await _refresh(user_id, _token_row(user_id))
            headers["Authorization"] = f"Bearer {token}"
            response = await client.post(_UPLOAD_URL, params=params, headers=headers, content=body)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("youtube returned an invalid upload response")
    return {
        "integration": "youtube",
        "operation": "upload_video",
        "video": {"id": result.get("id"), "status": result.get("status"), "snippet": result.get("snippet")},
        "secrets_exposed": False,
    }
