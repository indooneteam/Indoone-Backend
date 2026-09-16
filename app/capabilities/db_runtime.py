from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.capabilities import store

_DEFAULT_BUSY_TIMEOUT_MS = 5_000


def _configured_busy_timeout_ms() -> int:
    import os

    raw = os.getenv("INDOONE_SQLITE_BUSY_TIMEOUT_MS", str(_DEFAULT_BUSY_TIMEOUT_MS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_BUSY_TIMEOUT_MS
    return max(0, min(value, 60_000))


def configure_sqlite_runtime() -> None:
    """Apply safe SQLite connection pragmas to the capability-store runtime."""
    original_connect: Callable[[], sqlite3.Connection] = store._connect
    if getattr(original_connect, "_indoone_hardened", False):
        return

    def hardened_connect() -> sqlite3.Connection:
        connection = original_connect()
        connection.execute(f"PRAGMA busy_timeout={_configured_busy_timeout_ms()}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    setattr(hardened_connect, "_indoone_hardened", True)
    setattr(store, "_connect", hardened_connect)


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
