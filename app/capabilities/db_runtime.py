from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

from app.capabilities import store

_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_OAUTH_STATE_MAX_AGE_SECONDS = 600
_RUNTIME_LOCK = RLock()


def _configured_busy_timeout_ms() -> int:
    import os

    raw = os.getenv("INDOONE_SQLITE_BUSY_TIMEOUT_MS", str(_DEFAULT_BUSY_TIMEOUT_MS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_BUSY_TIMEOUT_MS
    return max(0, min(value, 60_000))


def _consume_oauth_state_atomic(state: str, integration_id: str, max_age_seconds: int = _OAUTH_STATE_MAX_AGE_SECONDS) -> dict[str, str] | None:
    """Consume one OAuth state under a write lock so replay cannot race a parallel callback."""
    max_age_seconds = max(0, int(max_age_seconds))
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        db.execute("DELETE FROM oauth_states WHERE created_at < ?", (cutoff,))
        row = db.execute(
            "SELECT user_id, integration_id, redirect_uri, created_at FROM oauth_states WHERE state = ? AND integration_id = ?",
            (state, integration_id),
        ).fetchone()
        if row is None:
            db.commit()
            return None

        db.execute("DELETE FROM oauth_states WHERE state = ? AND integration_id = ?", (state, integration_id))
        db.commit()

    try:
        created = datetime.fromisoformat(str(row["created_at"]))
    except (TypeError, ValueError):
        return None

    age = (datetime.now(timezone.utc) - created).total_seconds()
    if age > max_age_seconds:
        return None
    return {
        "user_id": str(row["user_id"]),
        "integration_id": str(row["integration_id"]),
        "redirect_uri": str(row["redirect_uri"]),
    }


def configure_sqlite_runtime() -> None:
    """Apply safe SQLite connection pragmas and concurrency guards to the capability-store runtime."""
    with _RUNTIME_LOCK:
        original_connect: Callable[[], sqlite3.Connection] = store._connect
        if not getattr(original_connect, "_indoone_hardened", False):

            def hardened_connect() -> sqlite3.Connection:
                connection = original_connect()
                connection.execute(f"PRAGMA busy_timeout={_configured_busy_timeout_ms()}")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                return connection

            setattr(hardened_connect, "_indoone_hardened", True)
            store._connect = hardened_connect

        original_initialize = store.initialize
        if not getattr(original_initialize, "_indoone_hardened", False):

            def hardened_initialize() -> None:
                with _RUNTIME_LOCK:
                    original_initialize()

            setattr(hardened_initialize, "_indoone_hardened", True)
            store.initialize = hardened_initialize

        if not getattr(store.consume_oauth_state, "_indoone_atomic", False):
            setattr(_consume_oauth_state_atomic, "_indoone_atomic", True)
            store.consume_oauth_state = _consume_oauth_state_atomic


def sqlite_runtime_status() -> dict[str, Any]:
    """Return a small, non-sensitive SQLite runtime health snapshot."""
    configure_sqlite_runtime()
    with store._connect() as db:
        journal_mode = str(db.execute("PRAGMA journal_mode").fetchone()[0])
        synchronous = int(db.execute("PRAGMA synchronous").fetchone()[0])
        busy_timeout = int(db.execute("PRAGMA busy_timeout").fetchone()[0])
        foreign_keys = int(db.execute("PRAGMA foreign_keys").fetchone()[0])
    return {
        "journal_mode": journal_mode,
        "synchronous": synchronous,
        "busy_timeout_ms": busy_timeout,
        "foreign_keys": bool(foreign_keys),
    }
