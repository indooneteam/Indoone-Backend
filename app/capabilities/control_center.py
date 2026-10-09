from __future__ import annotations

import hmac
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from app.capabilities import store

CHANNELS = ("whatsapp", "instagram", "telegram", "android")
EVENT_TYPES = ("request", "reply")
EVENT_STATUSES = ("success", "failed", "blocked", "skipped")
_INITIALIZED_DB_PATH: str | None = None

_REPLY_ENV_DEFAULTS = {
    "whatsapp": "INDOONE_WHATSAPP_AI_REPLY_ENABLED",
    "instagram": "INDOONE_INSTAGRAM_AI_REPLY_ENABLED",
    "telegram": "INDOONE_TELEGRAM_AI_REPLY_ENABLED",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def initialize_control_center() -> None:
    """Create Control Center tables once per configured capabilities database."""
    global _INITIALIZED_DB_PATH
    db_path = str(store._db_path())
    if _INITIALIZED_DB_PATH == db_path:
        return
    store.initialize()
    with closing(store._connect()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS control_center_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL CHECK(setting_value IN ('true', 'false')),
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_center_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('request', 'reply')),
                status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'blocked', 'skipped')),
                path TEXT NOT NULL,
                http_status INTEGER,
                dedupe_key TEXT UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_control_events_created
                ON control_center_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_control_events_channel_created
                ON control_center_events(channel, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_control_events_type_status
                ON control_center_events(event_type, status, created_at DESC);
            """
        )
        db.commit()
    _INITIALIZED_DB_PATH = db_path


def _setting_default(key: str) -> bool:
    if key == "global_intake_enabled" or key == "global_replies_enabled":
        return True
    for channel in CHANNELS:
        if key == f"{channel}_intake_enabled":
            return True
        if key == f"{channel}_reply_enabled":
            env_name = _REPLY_ENV_DEFAULTS.get(channel)
            # Android's existing /api/chat behavior replies by default.
            return _env_enabled(env_name, False) if env_name else True
    raise ValueError(f"unknown Control Center setting: {key}")


def get_setting(key: str) -> bool:
    """Read a persisted setting; use safe, backward-compatible defaults before first save."""
    if key not in {"global_intake_enabled", "global_replies_enabled"} and not any(
        key == f"{channel}_{suffix}" for channel in CHANNELS for suffix in ("intake_enabled", "reply_enabled")
    ):
        raise ValueError(f"unknown Control Center setting: {key}")
    initialize_control_center()
    with closing(store._connect()) as db:
        row = db.execute(
            "SELECT setting_value FROM control_center_settings WHERE setting_key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return _setting_default(key)
    return str(row["setting_value"]) == "true"


def _save_setting(db: sqlite3.Connection, key: str, value: bool) -> None:
    db.execute(
        """
        INSERT INTO control_center_settings(setting_key, setting_value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(setting_key) DO UPDATE SET
            setting_value=excluded.setting_value,
            updated_at=excluded.updated_at
        """,
        (key, "true" if value else "false", _now()),
    )


def get_control_state() -> dict[str, Any]:
    global_intake = get_setting("global_intake_enabled")
    global_replies = get_setting("global_replies_enabled")
    channels: dict[str, dict[str, bool]] = {}
    for channel in CHANNELS:
        intake = get_setting(f"{channel}_intake_enabled")
        reply = get_setting(f"{channel}_reply_enabled")
        channels[channel] = {
            "intake_enabled": intake,
            "effective_intake_enabled": global_intake and intake,
            "reply_enabled": reply,
            "effective_reply_enabled": global_replies and reply,
        }
    return {
        "status": "ok",
        "controls": {
            "global_intake_enabled": global_intake,
            "global_replies_enabled": global_replies,
            "channels": channels,
        },
    }


def update_control_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist only explicitly supplied booleans; unknown keys fail closed."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("at least one Control Center setting is required")
    allowed_globals = {"global_intake_enabled", "global_replies_enabled"}
    if set(payload) - allowed_globals - {"channels"}:
        raise ValueError("unknown Control Center setting")
    updates: dict[str, bool] = {}
    for key in allowed_globals.intersection(payload):
        if type(payload[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
        updates[key] = payload[key]

    if "channels" in payload:
        channel_payload = payload["channels"]
        if not isinstance(channel_payload, dict) or not channel_payload:
            raise ValueError("channels must be a non-empty object")
        for channel, flags in channel_payload.items():
            if channel not in CHANNELS or not isinstance(flags, dict) or not flags:
                raise ValueError("unknown channel or invalid channel settings")
            for flag, value in flags.items():
                if flag not in {"intake_enabled", "reply_enabled"}:
                    raise ValueError(f"unknown setting for {channel}")
                if type(value) is not bool:
                    raise ValueError(f"{channel}.{flag} must be a boolean")
                suffix = "intake_enabled" if flag == "intake_enabled" else "reply_enabled"
                updates[f"{channel}_{suffix}"] = value

    if not updates:
        raise ValueError("at least one Control Center setting is required")

    initialize_control_center()
    with closing(store._connect()) as db:
        for key, value in updates.items():
            _save_setting(db, key, value)
        db.commit()
    return get_control_state()


def intake_enabled(channel: str) -> bool:
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    return get_setting("global_intake_enabled") and get_setting(f"{channel}_intake_enabled")


def replies_enabled(channel: str) -> bool:
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    return get_setting("global_replies_enabled") and get_setting(f"{channel}_reply_enabled")


def is_control_center_admin_token(token: str) -> bool:
    """The admin API uses a dedicated server-side token, never an app/user token."""
    expected = os.getenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "").strip()
    supplied = token.strip()
    return bool(len(expected) >= 32 and supplied and hmac.compare_digest(expected, supplied))


def is_control_center_admin_authorization(authorization: str) -> bool:
    scheme, separator, token = authorization.partition(" ")
    return bool(separator and scheme.lower() == "bearer" and is_control_center_admin_token(token))


def record_control_event(
    channel: str,
    event_type: str,
    status: str,
    path: str,
    http_status: int | None = None,
    dedupe_key: str | None = None,
) -> bool:
    """Save a privacy-safe event; request/reply content and customer identifiers are never stored."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {event_type}")
    if status not in EVENT_STATUSES:
        raise ValueError(f"unknown event status: {status}")
    normalized_path = path.strip()[:256] or "/"
    key = dedupe_key.strip()[:512] if dedupe_key else None
    initialize_control_center()
    with closing(store._connect()) as db:
        cursor = db.execute(
            """
            INSERT OR IGNORE INTO control_center_events(
                channel, event_type, status, path, http_status, dedupe_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (channel, event_type, status, normalized_path, http_status, key, _now()),
        )
        db.commit()
        return cursor.rowcount > 0


def _empty_metrics() -> dict[str, Any]:
    return {
        "requests": {"total": 0, "success": 0, "failed": 0, "blocked": 0},
        "replies": {"sent": 0, "failed": 0, "skipped": 0},
    }


def get_control_metrics(window_hours: int = 24) -> dict[str, Any]:
    window_hours = max(1, min(int(window_hours), 720))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    initialize_control_center()
    channels: dict[str, dict[str, Any]] = {channel: _empty_metrics() for channel in CHANNELS}
    with closing(store._connect()) as db:
        rows = db.execute(
            """
            SELECT channel, event_type, status, COUNT(*) AS count
            FROM control_center_events
            WHERE created_at >= ?
            GROUP BY channel, event_type, status
            """,
            (cutoff,),
        ).fetchall()
    for row in rows:
        channel = str(row["channel"])
        event_type = str(row["event_type"])
        status = str(row["status"])
        count = int(row["count"])
        if channel not in channels:
            continue
        if event_type == "request":
            channels[channel]["requests"]["total"] += count
            if status in {"success", "failed", "blocked"}:
                channels[channel]["requests"][status] += count
        elif event_type == "reply":
            metric_key = {"success": "sent", "failed": "failed", "skipped": "skipped"}.get(status)
            if metric_key:
                channels[channel]["replies"][metric_key] += count

    totals = _empty_metrics()
    for data in channels.values():
        for key in totals["requests"]:
            totals["requests"][key] += data["requests"][key]
        for key in totals["replies"]:
            totals["replies"][key] += data["replies"][key]
    return {"status": "ok", "window_hours": window_hours, "channels": channels, "totals": totals}


def get_control_activity(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    initialize_control_center()
    with closing(store._connect()) as db:
        rows = db.execute(
            """
            SELECT channel, event_type, status, path, http_status, created_at
            FROM control_center_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {
        "status": "ok",
        "events": [
            {
                "channel": str(row["channel"]),
                "event_type": str(row["event_type"]),
                "status": str(row["status"]),
                "path": str(row["path"]),
                "http_status": int(row["http_status"]) if row["http_status"] is not None else None,
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ],
    }
