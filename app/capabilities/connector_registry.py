from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConnectorSpec:
    id: str
    name: str
    category: str
    auth: str
    read_capabilities: tuple[str, ...]
    write_capabilities: tuple[str, ...] = ()

    def capabilities(self) -> tuple[str, ...]:
        return self.read_capabilities + self.write_capabilities

    def allows(self, capability: str) -> bool:
        return capability.strip().lower() in {item.lower() for item in self.capabilities()}


CONNECTORS: tuple[ConnectorSpec, ...] = (
    ConnectorSpec("gmail", "Gmail", "communication", "oauth2", ("messages.search", "messages.read"), ("messages.send",)),
    ConnectorSpec("google_calendar", "Google Calendar", "productivity", "oauth2", ("events.read",), ("events.create", "events.update", "events.delete")),
    ConnectorSpec("google_drive", "Google Drive", "storage", "oauth2", ("files.search", "files.read"), ("files.upload",)),
    ConnectorSpec("google_photos", "Google Photos", "media", "oauth2", ("media.search", "media.read"), ("media.upload",)),
    ConnectorSpec("telegram", "Telegram", "communication", "bot", ("messages.read",), ("messages.send",)),
    ConnectorSpec("whatsapp", "WhatsApp", "communication", "oauth2", ("messages.read",), ("messages.send",)),
    ConnectorSpec("youtube", "YouTube", "media", "oauth2", ("search.read", "videos.read")),
    ConnectorSpec("instagram", "Instagram", "social", "oauth2", ("profile.read", "media.read"), ("media.publish", "messages.send")),
    ConnectorSpec("facebook", "Facebook", "social", "oauth2", ("profile.read", "pages.read"), ("posts.publish", "messages.send")),
    ConnectorSpec("canva", "Canva", "productivity", "oauth2", ("designs.read",), ("designs.write", "exports.create")),
    ConnectorSpec("github", "GitHub", "developer", "oauth2", ("repos.read", "issues.read", "pull_requests.read"), ("issues.write", "comments.write")),
)


def get_connector(connector_id: str) -> ConnectorSpec | None:
    key = connector_id.strip().lower()
    return next((item for item in CONNECTORS if item.id == key), None)


def list_connectors() -> list[dict[str, object]]:
    return [asdict(item) for item in CONNECTORS]


def allows_capability(connector_id: str, capability: str) -> bool:
    connector = get_connector(connector_id)
    return connector is not None and connector.allows(capability)
