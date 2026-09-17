from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token

_INTEGRATION_ID = "google_photos"
_OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_API_URL = "https://photoslibrary.googleapis.com/v1"
_UPLOAD_URL = "https://photoslibrary.googleapis.com/v1/uploads"
_READ_SCOPE = "https://www.googleapis.com/auth/photoslibrary.readonly"
_APPEND_SCOPE = "https://www.googleapis.com/auth/photoslibrary.appendonly"


def _credentials() -> tuple[str, str]:
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    return client_id, client_secret


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def build_google_photos_authorization(
    state: str,
    redirect_uri: str,
    include_upload: bool = True,
) -> dict[str, object]:
    client_id, _ = _credentials()
    scopes = [_READ_SCOPE]
    if include_upload:
        scopes.append(_APPEND_SCOPE)
    scope = " ".join(scopes)
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
        "integration": _INTEGRATION_ID,
        "authorization_url": f"{_OAUTH_URL}?{urlencode(params)}",
        "scope": scope,
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
        raise RuntimeError("stored google photos token cannot be decrypted") from exc


def _expires_at(expires_in: object) -> str | None:
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    return None


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("google photos oauth response contains no access token")
    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = fallback_refresh
    scope = str(body.get("scope") or f"{_READ_SCOPE} {_APPEND_SCOPE}")
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


async def exchange_google_photos_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _INTEGRATION_ID)
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    client_id, client_secret = _credentials()
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code.strip(),
        "redirect_uri": state_data["redirect_uri"],
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(_TOKEN_URL, data=payload, headers={"Accept": "application/json"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos oauth returned an invalid response")
    _store_token(state_data["user_id"], body)
    return {
        "integration": _INTEGRATION_ID,
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": str(body.get("scope") or f"{_READ_SCOPE} {_APPEND_SCOPE}"),
        "expires_at": _expires_at(body.get("expires_in")),
        "secrets_exposed": False,
    }


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if row is None:
        raise ValueError("google photos is not connected for user")
    return row


async def _refresh_access_token(user_id: str, row: dict[str, object]) -> str:
    refresh_token = _decode(row, "refresh_token")
    if not refresh_token:
        raise RuntimeError("google photos access token expired and no refresh token is stored")
    client_id, client_secret = _credentials()
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
        raise RuntimeError("google photos oauth refresh returned an invalid response")
    _store_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if refreshed is None:
        raise RuntimeError("refreshed google photos token could not be loaded")
    token = _decode(refreshed, "access_token")
    if not token:
        raise RuntimeError("refreshed google photos token is empty")
    return token


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decode(row, "access_token")
    if not token:
        raise RuntimeError("stored google photos access token is empty")
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(seconds=30):
                return await _refresh_access_token(user_id, row)
        except ValueError:
            pass
    return token


async def list_media(user_id: str, page_size: int = 25, page_token: str = "") -> dict[str, object]:
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    token = await _access_token(user_id)
    params: dict[str, object] = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_API_URL}/mediaItems",
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos media response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def search_media(
    user_id: str,
    page_size: int = 25,
    page_token: str = "",
    album_id: str = "",
    media_type: str = "",
    order: str = "",
) -> dict[str, object]:
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    token = await _access_token(user_id)
    body: dict[str, object] = {"pageSize": page_size}
    if page_token:
        body["pageToken"] = page_token
    if album_id:
        body["albumId"] = album_id.strip()
    if media_type:
        normalized = media_type.strip().upper()
        if normalized not in {"ALL_MEDIA", "PHOTO", "VIDEO"}:
            raise ValueError("media_type must be ALL_MEDIA, PHOTO, or VIDEO")
        body["filters"] = {"mediaTypeFilter": {"mediaTypes": [normalized]}}
    if order:
        normalized_order = order.strip().upper()
        if normalized_order not in {"MEDIA_ITEM_DATE_DESC", "MEDIA_ITEM_DATE_ASC"}:
            raise ValueError("order must be MEDIA_ITEM_DATE_DESC or MEDIA_ITEM_DATE_ASC")
        body["orderBy"] = normalized_order
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_API_URL}/mediaItems:search",
            json=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("google photos search response is invalid")
    return {"integration": _INTEGRATION_ID, **result, "secrets_exposed": False}


async def get_media_item(user_id: str, media_item_id: str) -> dict[str, object]:
    if not media_item_id.strip():
        raise ValueError("media_item_id is required")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_API_URL}/mediaItems/{media_item_id.strip()}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos media item response is invalid")
    return {"integration": _INTEGRATION_ID, "media_item": body, "secrets_exposed": False}


async def list_albums(user_id: str, page_size: int = 20, page_token: str = "") -> dict[str, object]:
    if not 1 <= page_size <= 50:
        raise ValueError("page_size must be between 1 and 50")
    token = await _access_token(user_id)
    params: dict[str, object] = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_API_URL}/albums",
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos albums response is invalid")
    return {"integration": _INTEGRATION_ID, **body, "secrets_exposed": False}


async def create_album(user_id: str, title: str, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("google photos album creation requires explicit approval")
    title = title.strip()
    if not title:
        raise ValueError("title is required")
    if len(title) > 500:
        raise ValueError("title must be 500 characters or fewer")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_API_URL}/albums",
            json={"album": {"title": title}},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos album response is invalid")
    return {"integration": _INTEGRATION_ID, "created": True, "album": body, "secrets_exposed": False}


async def upload_media(
    user_id: str,
    content: bytes,
    filename: str,
    mime_type: str,
    description: str = "",
    album_id: str = "",
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("google photos upload requires explicit approval")
    if not content:
        raise ValueError("content is required")
    if len(content) > 25 * 1024 * 1024:
        raise ValueError("request upload is limited to 25 MiB")
    if not mime_type.startswith(("image/", "video/")):
        raise ValueError("mime_type must be an image or video MIME type")
    token = await _access_token(user_id)
    upload_headers = {
        "Authorization": f"Bearer {token}",
        "Content-type": "application/octet-stream",
        "X-Goog-Upload-File-Name": filename.strip() or "upload",
        "X-Goog-Upload-Protocol": "raw",
        "X-Goog-Upload-Raw-Size": str(len(content)),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(_UPLOAD_URL, content=content, headers=upload_headers)
        response.raise_for_status()
        upload_token = response.text.strip().strip('"')
        if not upload_token:
            raise RuntimeError("google photos upload returned no upload token")
        creation: dict[str, object] = {
            "newMediaItems": [
                {
                    "description": description[:1000],
                    "simpleMediaItem": {"uploadToken": upload_token},
                }
            ]
        }
        if album_id.strip():
            creation["albumId"] = album_id.strip()
        response = await client.post(
            f"{_API_URL}/mediaItems:batchCreate",
            json=creation,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google photos batchCreate response is invalid")
    return {"integration": _INTEGRATION_ID, "uploaded": True, "result": body, "secrets_exposed": False}
