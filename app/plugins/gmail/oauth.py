"""Gmail-specific OAuth connection and token exchange operations."""

from __future__ import annotations

import httpx

from app.capabilities.integrations import (
    _PROBE_URLS,
    _TOKEN_URLS,
    _fernet,
    _secret,
    _token_fields,
)
from app.capabilities.store import get_integration_token, upsert_integration_token


def gmail_server_authorization_config() -> dict[str, object]:
    """Validate server-side OAuth prerequisites and expose only the public client ID."""
    client_id, _ = _secret("gmail")
    _fernet()
    return {"integration": "gmail", "server_client_id": client_id, "offline_access": True}


async def exchange_gmail_server_auth_code(
    user_id: str,
    server_auth_code: str,
    requested_scope: str = "",
) -> dict[str, object]:
    """Exchange an Android server authorization code for encrypted, refreshable Gmail credentials."""
    normalized_user = user_id.strip()
    code = server_auth_code.strip()
    if not normalized_user:
        raise ValueError("user_id is required")
    if not code:
        raise ValueError("server_auth_code is required")

    client_id, client_secret = _secret("gmail")
    cipher = _fernet()
    async with httpx.AsyncClient(timeout=20.0) as client:
        token_response = await client.post(
            _TOKEN_URLS["gmail"],
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        payload = token_response.json()

    if not isinstance(payload, dict):
        raise RuntimeError("Google returned an invalid authorization-code response")
    access_token, refresh_token, token_type, granted_scope, expires_at = _token_fields("gmail", payload)
    if not expires_at:
        raise RuntimeError("Google did not return an access-token expiry")

    existing = get_integration_token(normalized_user, "gmail")
    existing_refresh: str | None = None
    if existing and existing.get("refresh_token"):
        try:
            existing_refresh = cipher.decrypt(bytes(existing["refresh_token"])).decode("utf-8")
        except Exception as exc:
            raise RuntimeError("stored oauth refresh token cannot be decrypted") from exc
    if not refresh_token and not existing_refresh:
        raise RuntimeError(
            "Google did not issue a refresh token; reconnect Gmail and approve offline access"
        )

    async with httpx.AsyncClient(timeout=20.0) as client:
        profile_response = await client.get(
            _PROBE_URLS["gmail"],
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    if not isinstance(profile, dict):
        raise RuntimeError("Google returned an invalid Gmail profile response")
    email_address = str(profile.get("emailAddress") or "").strip()
    if not email_address:
        raise RuntimeError("Google returned no Gmail account identity")

    final_scope = granted_scope or requested_scope.strip() or "https://www.googleapis.com/auth/gmail.modify"
    upsert_integration_token(
        normalized_user,
        "gmail",
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        token_type,
        final_scope,
        expires_at,
    )
    return {
        "integration": "gmail",
        "user_id": normalized_user,
        "connected": True,
        "provider_ok": True,
        "scope": final_scope,
        "expires_at": expires_at,
        "summary": {"email_address": email_address},
        "offline_access": True,
        "secrets_exposed": False,
    }


async def store_native_access_token(
    user_id: str,
    integration_id: str,
    access_token: str,
    scope: str = "",
    expires_at: str | None = None,
) -> dict[str, object]:
    normalized = integration_id.strip().lower()
    normalized_user = user_id.strip()
    token = access_token.strip()
    if normalized != "gmail":
        raise ValueError("native access-token connection is only supported for Gmail")
    if not normalized_user:
        raise ValueError("user_id is required")
    if not token:
        raise ValueError("access_token is required")

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(_PROBE_URLS["gmail"], headers=headers)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("gmail provider returned invalid profile response")
    email_address = str(body.get("emailAddress") or "").strip()
    if not email_address:
        raise RuntimeError("gmail provider returned no account identity")

    cipher = _fernet()
    normalized_scope = scope.strip() or "https://www.googleapis.com/auth/gmail.modify"
    upsert_integration_token(
        normalized_user,
        "gmail",
        cipher.encrypt(token.encode("utf-8")),
        None,
        "Bearer",
        normalized_scope,
        expires_at.strip() if isinstance(expires_at, str) and expires_at.strip() else None,
    )
    return {
        "integration": "gmail",
        "user_id": normalized_user,
        "connected": True,
        "provider_ok": True,
        "summary": {"email_address": email_address},
        "secrets_exposed": False,
    }
