from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token


@dataclass(frozen=True)
class Integration:
    id: str
    name: str
    category: str
    status: str
    auth_type: str
    permissions: tuple[str, ...]
    authorization_url: str = ""
    configured: bool = False


INTEGRATIONS: tuple[Integration, ...] = (
    Integration("github", "GitHub", "developer", "ready", "oauth2", ("repo:read", "issues:read", "pull_requests:read"), "https://github.com/login/oauth/authorize", bool(os.getenv("INDOONE_GITHUB_CLIENT_ID"))),
    Integration("google_calendar", "Google Calendar", "productivity", "ready", "oauth2", ("calendar:read", "calendar:write"), "https://accounts.google.com/o/oauth2/v2/auth", bool(os.getenv("INDOONE_GOOGLE_CLIENT_ID"))),
    Integration("slack", "Slack", "communication", "ready", "oauth2", ("channels:read", "messages:read", "messages:write"), "https://slack.com/oauth/v2/authorize", bool(os.getenv("INDOONE_SLACK_CLIENT_ID"))),
)

_CLIENT_ID_ENV = {"github": "INDOONE_GITHUB_CLIENT_ID", "google_calendar": "INDOONE_GOOGLE_CLIENT_ID", "slack": "INDOONE_SLACK_CLIENT_ID"}
_CLIENT_SECRET_ENV = {"github": "INDOONE_GITHUB_CLIENT_SECRET", "google_calendar": "INDOONE_GOOGLE_CLIENT_SECRET", "slack": "INDOONE_SLACK_CLIENT_SECRET"}
_TOKEN_URLS = {"github": "https://github.com/login/oauth/access_token", "google_calendar": "https://oauth2.googleapis.com/token", "slack": "https://slack.com/api/oauth.v2.access"}
_PROBE_URLS = {"github": "https://api.github.com/user", "google_calendar": "https://www.googleapis.com/calendar/v3/users/me/calendarList", "slack": "https://slack.com/api/auth.test"}


def list_integrations() -> list[dict[str, object]]:
    return [asdict(item) for item in INTEGRATIONS]


def get_integration(integration_id: str) -> dict[str, object] | None:
    normalized = integration_id.strip().lower()
    for integration in INTEGRATIONS:
        if integration.id == normalized:
            return asdict(integration)
    return None


def build_oauth_authorization(integration_id: str, state: str, redirect_uri: str) -> dict[str, object]:
    normalized = integration_id.strip().lower()
    integration = next((item for item in INTEGRATIONS if item.id == normalized), None)
    if integration is None:
        raise ValueError("integration not found")
    if integration.auth_type != "oauth2":
        raise ValueError("integration does not support oauth2")
    if not state.strip():
        raise ValueError("state is required")
    if not redirect_uri.strip():
        raise ValueError("redirect_uri is required")
    client_id_env = _CLIENT_ID_ENV.get(normalized)
    client_id = os.getenv(client_id_env, "") if client_id_env else ""
    if not client_id:
        raise RuntimeError("integration client id is not configured")
    params = {"client_id": client_id, "redirect_uri": redirect_uri.strip(), "state": state.strip()}
    if normalized == "github":
        params["scope"] = " ".join(integration.permissions)
    elif normalized == "google_calendar":
        params.update({"response_type": "code", "access_type": "offline", "scope": "https://www.googleapis.com/auth/calendar"})
    elif normalized == "slack":
        params["scope"] = ",".join(integration.permissions)
    return {"integration": integration.id, "authorization_url": f"{integration.authorization_url}?{urlencode(params)}", "state_required": True, "token_storage": "user-scoped-encrypted-server-side", "secrets_exposed": False}


def _fernet() -> Fernet:
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("oauth encryption key is invalid") from exc


def _secret(integration_id: str) -> tuple[str, str]:
    client_id = os.getenv(_CLIENT_ID_ENV[integration_id], "")
    client_secret = os.getenv(_CLIENT_SECRET_ENV[integration_id], "")
    if not client_id or not client_secret:
        raise RuntimeError("integration client credentials are not configured")
    return client_id, client_secret


def _token_fields(integration_id: str, payload: dict[str, object]) -> tuple[str, str | None, str, str, str | None]:
    if integration_id == "slack" and payload.get("ok") is False:
        raise RuntimeError(str(payload.get("error") or "oauth token exchange failed"))
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("oauth provider returned no access token")
    refresh = payload.get("refresh_token")
    refresh_token = refresh if isinstance(refresh, str) and refresh else None
    token_type = str(payload.get("token_type") or "Bearer")
    scope_value = payload.get("scope")
    scope = str(scope_value or "")
    expires_at: str | None = None
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    return access_token, refresh_token, token_type, scope, expires_at


async def exchange_oauth_code(integration_id: str, state: str, code: str) -> dict[str, object]:
    normalized = integration_id.strip().lower()
    integration = next((item for item in INTEGRATIONS if item.id == normalized), None)
    if integration is None:
        raise ValueError("integration not found")
    if not code.strip() or not state.strip():
        raise ValueError("code and state are required")
    state_data = consume_oauth_state(state.strip(), normalized)
    if state_data is None:
        raise ValueError("invalid or expired oauth state")
    client_id, client_secret = _secret(normalized)
    payload = {"client_id": client_id, "client_secret": client_secret, "code": code.strip(), "redirect_uri": state_data["redirect_uri"]}
    if normalized == "google_calendar":
        payload["grant_type"] = "authorization_code"
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(_TOKEN_URLS[normalized], data=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    access_token, refresh_token, token_type, scope, expires_at = _token_fields(normalized, body if isinstance(body, dict) else {})
    cipher = _fernet()
    upsert_integration_token(
        state_data["user_id"], normalized, cipher.encrypt(access_token.encode()),
        cipher.encrypt(refresh_token.encode()) if refresh_token else None,
        token_type, scope, expires_at,
    )
    return {"integration": normalized, "user_id": state_data["user_id"], "connected": True, "scope": scope, "expires_at": expires_at, "secrets_exposed": False}


def _provider_probe_summary(integration_id: str, payload: dict[str, object]) -> dict[str, object]:
    if integration_id == "github":
        return {"login": payload.get("login"), "name": payload.get("name"), "id": payload.get("id")}
    if integration_id == "google_calendar":
        items = payload.get("items")
        return {"calendar_count": len(items) if isinstance(items, list) else 0}
    if integration_id == "slack":
        return {"team": payload.get("team"), "user": payload.get("user"), "team_id": payload.get("team_id"), "user_id": payload.get("user_id")}
    return {}


async def probe_integration(user_id: str, integration_id: str) -> dict[str, object]:
    normalized = integration_id.strip().lower()
    if not user_id.strip():
        raise ValueError("user_id is required")
    if normalized not in _PROBE_URLS:
        raise ValueError("integration not found")
    token_row = get_integration_token(user_id.strip(), normalized)
    if token_row is None:
        raise ValueError("integration is not connected for user")
    cipher = _fernet()
    try:
        access_token = cipher.decrypt(bytes(token_row["access_token"])).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored oauth token cannot be decrypted") from exc
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(_PROBE_URLS[normalized], headers=headers)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("integration provider returned invalid response")
    if normalized == "slack" and body.get("ok") is False:
        raise RuntimeError(str(body.get("error") or "slack integration probe failed"))
    return {"integration": normalized, "user_id": user_id.strip(), "connected": True, "provider_ok": True, "summary": _provider_probe_summary(normalized, body), "secrets_exposed": False}
