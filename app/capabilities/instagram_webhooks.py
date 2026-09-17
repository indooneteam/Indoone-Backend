from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

_MAX_ENTRIES = 100
_MAX_EVENTS_PER_ENTRY = 100


def _verify_token() -> str:
    token = os.getenv("INDOONE_INSTAGRAM_WEBHOOK_VERIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("instagram webhook verify token is not configured")
    return token


def _app_secret() -> str:
    secret = os.getenv("INDOONE_INSTAGRAM_APP_SECRET", "").strip()
    if not secret:
        raise RuntimeError("instagram app credentials are not configured")
    return secret


def verify_challenge(mode: str, challenge: str, verify_token: str) -> str:
    if mode != "subscribe":
        raise ValueError("invalid webhook mode")
    expected = _verify_token()
    if not hmac.compare_digest(verify_token, expected):
        raise PermissionError("instagram webhook verification failed")
    challenge_value = challenge.strip()
    if not challenge_value:
        raise ValueError("webhook challenge is required")
    return challenge_value


def verify_signature(raw_body: bytes, signature: str) -> None:
    value = signature.strip()
    if not value.startswith("sha256="):
        raise PermissionError("instagram webhook signature is invalid")
    provided = value[7:]
    expected = hmac.new(_app_secret().encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided, expected):
        raise PermissionError("instagram webhook signature is invalid")


def normalize_event(payload: dict[str, Any]) -> dict[str, object]:
    if str(payload.get("object", "")).strip().lower() != "instagram":
        raise ValueError("instagram webhook object is invalid")

    raw_entries = payload.get("entry", [])
    if not isinstance(raw_entries, list):
        raise ValueError("instagram webhook entries must be a list")
    if len(raw_entries) > _MAX_ENTRIES:
        raise ValueError("instagram webhook contains too many entries")

    normalized_entries: list[dict[str, object]] = []
    event_count = 0

    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes", [])
        messaging = entry.get("messaging", [])
        change_items = changes if isinstance(changes, list) else []
        messaging_items = messaging if isinstance(messaging, list) else []
        if len(change_items) > _MAX_EVENTS_PER_ENTRY or len(messaging_items) > _MAX_EVENTS_PER_ENTRY:
            raise ValueError("instagram webhook entry contains too many events")
        event_count += len(change_items) + len(messaging_items)
        normalized_entries.append(
            {
                "id": entry.get("id"),
                "time": entry.get("time"),
                "changes": change_items,
                "messaging": messaging_items,
            }
        )

    event_id_source = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    event_id = hashlib.sha256(event_id_source).hexdigest()
    return {
        "integration": "instagram",
        "object": "instagram",
        "event_id": event_id,
        "event_count": event_count,
        "entries": normalized_entries,
        "secrets_exposed": False,
    }
