from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.capabilities.connector_registry import get_connector
from app.capabilities.store import get_integration_token_metadata


@dataclass(frozen=True)
class ConnectorAuthorization:
    allowed: bool
    reason: str = ""


_TOOL_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "gmail_search": ("gmail", "messages.search"),
    "gmail_read": ("gmail", "messages.read"),
}

# Provider scopes are translated to the normalized connector capabilities exposed
# by connector_registry. Exact capability names remain valid for future providers.
_SCOPE_ALIASES: dict[str, dict[str, frozenset[str]]] = {
    "gmail": {
        "gmail.modify": frozenset(("messages.search", "messages.read", "messages.send")),
        "gmail.readonly": frozenset(("messages.search", "messages.read")),
    }
}


def connector_requirement_for_tool(tool: str) -> tuple[str, str] | None:
    return _TOOL_REQUIREMENTS.get(tool.strip().lower())


def _scope_tokens(scope: str) -> tuple[str, ...]:
    return tuple(token for token in scope.replace(",", " ").split() if token)


def scope_allows(connector_id: str, scope: str, capability: str) -> bool:
    normalized_connector = connector_id.strip().lower()
    normalized_capability = capability.strip().lower()
    if not normalized_capability:
        return False
    for token in _scope_tokens(scope):
        token_key = token.casefold()
        if token_key == normalized_capability:
            return True
        aliases = _SCOPE_ALIASES.get(normalized_connector, {})
        if normalized_capability in aliases.get(token_key, frozenset()):
            return True
    return False


def _token_is_expired(expires_at: object) -> bool:
    if not expires_at:
        return False
    try:
        value = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value <= datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def authorize_connector(user_id: str, connector_id: str, capability: str) -> ConnectorAuthorization:
    normalized_user = user_id.strip()
    normalized_connector = connector_id.strip().lower()
    normalized_capability = capability.strip().lower()
    if not normalized_user:
        return ConnectorAuthorization(False, "authenticated user is required")
    if not normalized_connector or not normalized_capability:
        return ConnectorAuthorization(False, "connector and capability are required")

    connector = get_connector(normalized_connector)
    if connector is None:
        return ConnectorAuthorization(False, "connector is not registered")
    if not connector.allows(normalized_capability):
        return ConnectorAuthorization(False, "connector capability is not registered")

    metadata = get_integration_token_metadata(normalized_user, normalized_connector)
    if metadata is None:
        return ConnectorAuthorization(False, "connector is not connected for this user")
    if _token_is_expired(metadata.get("expires_at")):
        return ConnectorAuthorization(False, "connector authorization has expired")
    if not scope_allows(normalized_connector, str(metadata.get("scope", "")), normalized_capability):
        return ConnectorAuthorization(False, "connected token scope does not allow this capability")
    return ConnectorAuthorization(True, "connector capability authorized")


def authorize_tool(user_id: str, tool: str) -> ConnectorAuthorization:
    requirement = connector_requirement_for_tool(tool)
    if requirement is None:
        return ConnectorAuthorization(True, "tool does not require a connector")
    connector_id, capability = requirement
    return authorize_connector(user_id, connector_id, capability)
