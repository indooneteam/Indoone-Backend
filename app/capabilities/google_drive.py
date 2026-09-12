from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import (
    consume_oauth_state,
    get_integration_token,
    upsert_integration_token,
)

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_DRIVE_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3/files"
_OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/drive"
_MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def _client_credentials() -> tuple[str, str]:
    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    return client_id, client_secret


def build_google_drive_authorization(state: str, redirect_uri: str) -> dict[str, object]:
    client_id, _ = _client_credentials()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "scope": _SCOPE,
        "state": state.strip(),
    }
    return {
        "integration": "google_drive",
        "authorization_url": f"{_OAUTH_URL}?{urlencode(params)}",
        "scope": _SCOPE,
        "state_required": True,
        "secrets_exposed": False,
    }


def _decode_token(row: dict[str, object], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored oauth token cannot be decrypted") from exc


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("google oauth token response contains no access token")
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
        "google_drive",
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        str(body.get("token_type") or "Bearer"),
        str(body.get("scope") or _SCOPE),
        expires_at,
    )


async def exchange_google_drive_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), "google_drive")
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    client_id, client_secret = _client_credentials()
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
        raise RuntimeError("google oauth returned an invalid response")
    _store_token(state_data["user_id"], body)
    return {
        "integration": "google_drive",
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": str(body.get("scope") or _SCOPE),
        "expires_at": _expires_at_from_response(body),
        "secrets_exposed": False,
    }


def _expires_at_from_response(body: dict[str, object]) -> str | None:
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    return None


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), "google_drive")
    if row is None:
        raise ValueError("integration is not connected for user")
    return row


async def _refresh_access_token(user_id: str, row: dict[str, object]) -> str:
    refresh_token = _decode_token(row, "refresh_token")
    if not refresh_token:
        raise RuntimeError("google drive access token expired and no refresh token is stored")
    client_id, client_secret = _client_credentials()
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
        raise RuntimeError("google oauth refresh returned an invalid response")
    _store_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = get_integration_token(user_id.strip(), "google_drive")
    if refreshed is None:
        raise RuntimeError("refreshed google drive token could not be loaded")
    token = _decode_token(refreshed, "access_token")
    if not token:
        raise RuntimeError("refreshed google drive token is empty")
    return token


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    token = _decode_token(row, "access_token")
    if not token:
        raise RuntimeError("stored google drive access token is empty")
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


async def probe_google_drive(user_id: str) -> dict[str, object]:
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_DRIVE_API}/about", headers=_headers(token), params={"fields": "user(displayName,emailAddress),storageQuota"})
        if response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.get(f"{_DRIVE_API}/about", headers=_headers(token), params={"fields": "user(displayName,emailAddress),storageQuota"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google drive returned an invalid probe response")
    return {
        "integration": "google_drive",
        "user_id": user_id.strip(),
        "connected": True,
        "provider_ok": True,
        "summary": body,
        "secrets_exposed": False,
    }


async def list_drive_files(
    user_id: str,
    query: str = "",
    page_token: str = "",
    page_size: int = 50,
) -> dict[str, object]:
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")
    token = await _access_token(user_id)
    params: dict[str, object] = {
        "pageSize": page_size,
        "spaces": "drive",
        "orderBy": "modifiedTime desc",
        "fields": "nextPageToken,incompleteSearch,files(id,name,mimeType,size,modifiedTime,createdTime,webViewLink,parents,trashed,description)",
    }
    if query.strip():
        safe_query = query.replace("'", "\\'")
        params["q"] = f"trashed = false and name contains '{safe_query}'"
    else:
        params["q"] = "trashed = false"
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_DRIVE_API}/files", headers=_headers(token), params=params)
        if response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.get(f"{_DRIVE_API}/files", headers=_headers(token), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google drive returned an invalid file list")
    return {
        "integration": "google_drive",
        "user_id": user_id.strip(),
        "files": body.get("files") if isinstance(body.get("files"), list) else [],
        "next_page_token": body.get("nextPageToken"),
        "incomplete_search": bool(body.get("incompleteSearch", False)),
        "secrets_exposed": False,
    }


async def get_drive_file(user_id: str, file_id: str, download: bool = False, export_mime_type: str = "") -> dict[str, object]:
    file_id = file_id.strip()
    if not file_id or len(file_id) > 512:
        raise ValueError("file_id is required and must be at most 512 characters")
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        meta_response = await client.get(
            f"{_DRIVE_API}/files/{file_id}",
            headers=_headers(token),
            params={"fields": "id,name,mimeType,size,modifiedTime,createdTime,webViewLink,parents,trashed,description"},
        )
        if meta_response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            meta_response = await client.get(
                f"{_DRIVE_API}/files/{file_id}",
                headers=_headers(token),
                params={"fields": "id,name,mimeType,size,modifiedTime,createdTime,webViewLink,parents,trashed,description"},
            )
        meta_response.raise_for_status()
        metadata = meta_response.json()
        if not isinstance(metadata, dict):
            raise RuntimeError("google drive returned invalid file metadata")
        if not download:
            return {"integration": "google_drive", "user_id": user_id.strip(), "file": metadata, "secrets_exposed": False}
        mime_type = str(metadata.get("mimeType") or "")
        if export_mime_type.strip():
            response = await client.get(
                f"{_DRIVE_API}/files/{file_id}/export",
                headers={"Authorization": f"Bearer {token}"},
                params={"mimeType": export_mime_type.strip()},
            )
        elif mime_type.startswith("application/vnd.google-apps"):
            raise ValueError("google-native files require export_mime_type for download")
        else:
            response = await client.get(
                f"{_DRIVE_API}/files/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"alt": "media"},
            )
        if response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            if export_mime_type.strip():
                response = await client.get(
                    f"{_DRIVE_API}/files/{file_id}/export",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"mimeType": export_mime_type.strip()},
                )
            else:
                response = await client.get(
                    f"{_DRIVE_API}/files/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"alt": "media"},
                )
        response.raise_for_status()
        if len(response.content) > _MAX_DOWNLOAD_BYTES:
            raise ValueError("file is too large to return through the API; use a direct signed download flow instead")
        return {
            "integration": "google_drive",
            "user_id": user_id.strip(),
            "file": metadata,
            "download": {
                "mime_type": response.headers.get("content-type", mime_type),
                "size": len(response.content),
                "content_base64": base64.b64encode(response.content).decode("ascii"),
            },
            "secrets_exposed": False,
        }


async def upload_drive_file(
    user_id: str,
    filename: str,
    mime_type: str,
    content: bytes,
    parent_id: str = "",
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for google drive upload operations")
    if not filename.strip():
        raise ValueError("filename is required")
    if not mime_type.strip():
        raise ValueError("mime_type is required")
    if not content:
        raise ValueError("content is required")
    if len(content) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("upload exceeds the 15 MB API limit")
    token = await _access_token(user_id)
    boundary = "indoone-drive-boundary"
    metadata: dict[str, object] = {"name": filename.strip()}
    if parent_id.strip():
        metadata["parents"] = [parent_id.strip()]
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    body = (
        b"--" + boundary.encode() + b"\r\n"
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        + metadata_bytes
        + b"\r\n--"
        + boundary.encode()
        + b"\r\n"
        + f"Content-Type: {mime_type.strip()}\r\n\r\n".encode("utf-8")
        + content
        + b"\r\n--"
        + boundary.encode()
        + b"--\r\n"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/related; boundary={boundary}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{_DRIVE_UPLOAD_API}?uploadType=multipart", headers=headers, content=body)
        if response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            headers["Authorization"] = f"Bearer {token}"
            response = await client.post(f"{_DRIVE_UPLOAD_API}?uploadType=multipart", headers=headers, content=body)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("google drive returned an invalid upload response")
    return {
        "integration": "google_drive",
        "user_id": user_id.strip(),
        "operation": "upload",
        "file": result,
        "secrets_exposed": False,
    }
