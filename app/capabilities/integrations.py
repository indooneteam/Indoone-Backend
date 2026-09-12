from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from urllib.parse import urlencode


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
    Integration(
        "github",
        "GitHub",
        "developer",
        "ready",
        "oauth2",
        ("repo:read", "issues:read", "pull_requests:read"),
        "https://github.com/login/oauth/authorize",
        bool(os.getenv("INDOONE_GITHUB_CLIENT_ID")),
    ),
    Integration(
        "google_calendar",
        "Google Calendar",
        "productivity",
        "ready",
        "oauth2",
        ("calendar:read", "calendar:write"),
        "https://accounts.google.com/o/oauth2/v2/auth",
        bool(os.getenv("INDOONE_GOOGLE_CLIENT_ID")),
    ),
    Integration(
        "slack",
        "Slack",
        "communication",
        "ready",
        "oauth2",
        ("channels:read", "messages:read", "messages:write"),
        "https://slack.com/oauth/v2/authorize",
        bool(os.getenv("INDOONE_SLACK_CLIENT_ID")),
    ),
)


_CLIENT_ID_ENV = {
    "github": "INDOONE_GITHUB_CLIENT_ID",
    "google_calendar": "INDOONE_GOOGLE_CLIENT_ID",
    "slack": "INDOONE_SLACK_CLIENT_ID",
}


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
    return {
        "integration": integration.id,
        "authorization_url": f"{integration.authorization_url}?{urlencode(params)}",
        "state_required": True,
        "token_storage": "user-scoped-server-side",
        "secrets_exposed": False,
    }
