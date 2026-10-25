"""Server-side Zoho Mail OAuth flow and encrypted token persistence."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from urllib.parse import urlencode, urlsplit

import httpx

from app.capabilities.connectors.integrations import _fernet
from app.capabilities.connectors.store import (
    consume_oauth_state,
    delete_integration_token,
    create_oauth_state,
    get_integration_token,
    upsert_integration_token,
)

_DEFAULT_ACCOUNTS_BASE_URL = "https://accounts.zoho.in"
_DEFAULT_MAIL_API_BASE_URL = "https://mail.zoho.in/api"
ZOHO_MAIL_REQUIRED_SCOPES = (
    "ZohoMail.accounts.READ",
    "ZohoMail.folders.READ",
    "ZohoMail.messages.READ",
    "ZohoMail.messages.CREATE",
    "ZohoMail.messages.DELETE",
)
_DEFAULT_SCOPE = ",".join(ZOHO_MAIL_REQUIRED_SCOPES)


def _https_url(value: str, setting_name: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError(f"{setting_name} must be a valid HTTPS URL without credentials, query, or fragment.")
    return normalized


def zoho_mail_auth_config() -> dict[str, object]:
    accounts_base = _https_url(
        os.getenv("INDOONE_ZOHO_ACCOUNTS_BASE_URL", _DEFAULT_ACCOUNTS_BASE_URL),
        "INDOONE_ZOHO_ACCOUNTS_BASE_URL",
    )
    mail_base = _https_url(
        os.getenv("INDOONE_ZOHO_MAIL_API_BASE_URL", _DEFAULT_MAIL_API_BASE_URL),
        "INDOONE_ZOHO_MAIL_API_BASE_URL",
    )
    client_id = os.getenv("INDOONE_ZOHO_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_ZOHO_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("INDOONE_ZOHO_REDIRECT_URI", "").strip()
    return {
        "integration": "zoho_mail",
        "configured": bool(client_id and client_secret and redirect_uri),
        "client_id_configured": bool(client_id),
        "client_secret_configured": bool(client_secret),
        "redirect_uri_configured": bool(redirect_uri),
        "accounts_base_url": accounts_base,
        "mail_api_base_url": mail_base,
        "scope": _DEFAULT_SCOPE,
        "secrets_exposed": False,
    }


def _required_settings() -> tuple[str, str, str, str, str]:
    client_id = os.getenv("INDOONE_ZOHO_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_ZOHO_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("INDOONE_ZOHO_REDIRECT_URI", "").strip()
    accounts_base = _https_url(
        os.getenv("INDOONE_ZOHO_ACCOUNTS_BASE_URL", _DEFAULT_ACCOUNTS_BASE_URL),
        "INDOONE_ZOHO_ACCOUNTS_BASE_URL",
    )
    # Always request the least-privilege set needed for mailbox/folder reads,
    # draft/send, and deletion. A stale environment override must not silently
    # omit a permission required by the integration.
    scope = _DEFAULT_SCOPE
    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            "Zoho Mail OAuth is not configured. Set INDOONE_ZOHO_CLIENT_ID, "
            "INDOONE_ZOHO_CLIENT_SECRET, and INDOONE_ZOHO_REDIRECT_URI on the backend."
        )
    parsed_redirect = urlsplit(redirect_uri)
    if parsed_redirect.scheme != "https" or not parsed_redirect.netloc or parsed_redirect.username or parsed_redirect.password:
        raise RuntimeError("INDOONE_ZOHO_REDIRECT_URI must be a valid HTTPS callback URL.")
    return client_id, client_secret, redirect_uri, accounts_base, scope


async def disconnect_zoho_mail(user_id: str) -> dict[str, object]:
    """Revoke the user's Zoho grant when possible, then remove local credentials."""
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")

    row = get_integration_token(normalized_user_id, "zoho_mail")
    if row is None:
        return {
            "integration": "zoho_mail",
            "user_id": normalized_user_id,
            "connected": False,
            "remote_revoked": True,
            "secrets_exposed": False,
        }

    remote_revoked = False
    token_value = ""
    token_type = "refresh_token"
    try:
        cipher = _fernet()
        encrypted_refresh = row.get("refresh_token")
        if encrypted_refresh:
            token_value = cipher.decrypt(bytes(encrypted_refresh)).decode("utf-8")
        else:
            token_type = "access_token"
            token_value = cipher.decrypt(bytes(row["access_token"])).decode("utf-8")
    except Exception:
        # Even if an old token cannot be decrypted, local access must be removed.
        token_value = ""

    client_id = os.getenv("INDOONE_ZOHO_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_ZOHO_CLIENT_SECRET", "").strip()
    if token_value and client_id and client_secret:
        accounts_base = _https_url(
            os.getenv("INDOONE_ZOHO_ACCOUNTS_BASE_URL", _DEFAULT_ACCOUNTS_BASE_URL),
            "INDOONE_ZOHO_ACCOUNTS_BASE_URL",
        )
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{accounts_base}/oauth/v2/revoke/token",
                    data={"token": token_value, "token_type": token_type},
                    headers={
                        "Authorization": f"Basic {credentials}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                remote_revoked = True
        except httpx.HTTPError:
            # Local disconnect should still complete if Zoho is unavailable or
            # reports a token that has already expired/revoked. Never log tokens.
            logging.getLogger("indoone.zoho_mail").warning(
                "Zoho token revocation failed; local credentials will still be removed"
            )

    delete_integration_token(normalized_user_id, "zoho_mail")
    return {
        "integration": "zoho_mail",
        "user_id": normalized_user_id,
        "connected": False,
        "remote_revoked": remote_revoked,
        "secrets_exposed": False,
    }


def start_zoho_mail_authorization(user_id: str) -> dict[str, object]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")
    client_id, _, redirect_uri, accounts_base, scope = _required_settings()
    state = token_urlsafe(32)
    create_oauth_state(state, normalized_user_id, "zoho_mail", redirect_uri)
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return {
        "integration": "zoho_mail",
        "authorization_url": f"{accounts_base}/oauth/v2/auth?{urlencode(params)}",
        "state_required": True,
        "token_storage": "user-scoped-encrypted-server-side",
        "secrets_exposed": False,
    }


def cancel_zoho_mail_authorization(state: str) -> bool:
    normalized_state = state.strip()
    if not normalized_state:
        return False
    return consume_oauth_state(normalized_state, "zoho_mail") is not None


async def exchange_zoho_mail_authorization_code(code: str, state: str) -> dict[str, object]:
    normalized_code = code.strip()
    normalized_state = state.strip()
    if not normalized_code or not normalized_state:
        raise ValueError("code and state are required")
    state_data = consume_oauth_state(normalized_state, "zoho_mail")
    if state_data is None:
        raise ValueError("Zoho Mail authorization state is invalid or expired")

    client_id, client_secret, redirect_uri, accounts_base, requested_scope = _required_settings()
    if state_data["redirect_uri"] != redirect_uri:
        raise ValueError("Zoho Mail callback URI does not match the authorization request")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{accounts_base}/oauth/v2/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "code": normalized_code,
                "scope": requested_scope,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError("Zoho returned an invalid OAuth token response")
    if payload.get("error"):
        raise RuntimeError("Zoho rejected the authorization code")
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("Zoho returned no access token")
    refresh_value = payload.get("refresh_token")
    refresh_token = refresh_value.strip() if isinstance(refresh_value, str) and refresh_value.strip() else None
    existing = get_integration_token(state_data["user_id"], "zoho_mail")
    if not refresh_token and not (existing and existing.get("refresh_token")):
        raise RuntimeError("Zoho returned no refresh token. Reconnect and grant offline access.")

    expires_in = payload.get("expires_in", 3600)
    if isinstance(expires_in, str):
        try:
            expires_in = int(expires_in)
        except ValueError:
            expires_in = 3600
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        expires_in = 3600
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    granted_scope = str(payload.get("scope") or requested_scope)
    cipher = _fernet()
    upsert_integration_token(
        state_data["user_id"],
        "zoho_mail",
        cipher.encrypt(access_token.encode("utf-8")),
        cipher.encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        str(payload.get("token_type") or "Zoho-oauthtoken"),
        granted_scope,
        expires_at,
    )
    return {
        "integration": "zoho_mail",
        "user_id": state_data["user_id"],
        "connected": True,
        "scope": granted_scope,
        "expires_at": expires_at,
        "secrets_exposed": False,
    }
