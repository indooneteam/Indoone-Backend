from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Integration:
    id: str
    name: str
    category: str
    status: str
    auth_type: str
    permissions: tuple[str, ...]
    configured: bool = False


INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        "github",
        "GitHub",
        "developer",
        "planned",
        "oauth2",
        ("repo:read", "issues:read", "pull_requests:read"),
    ),
    Integration(
        "google_calendar",
        "Google Calendar",
        "productivity",
        "planned",
        "oauth2",
        ("calendar:read", "calendar:write"),
    ),
    Integration(
        "slack",
        "Slack",
        "communication",
        "planned",
        "oauth2",
        ("channels:read", "messages:read", "messages:write"),
    ),
)


def list_integrations() -> list[dict[str, object]]:
    return [asdict(item) for item in INTEGRATIONS]


def get_integration(integration_id: str) -> dict[str, object] | None:
    normalized = integration_id.strip().lower()
    for integration in INTEGRATIONS:
        if integration.id == normalized:
            return asdict(integration)
    return None
