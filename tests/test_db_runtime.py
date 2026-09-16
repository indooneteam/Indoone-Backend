from __future__ import annotations

import sqlite3

from app.capabilities import db_runtime, store


def test_sqlite_runtime_applies_safe_pragmas(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    store.initialize()

    status = db_runtime.sqlite_runtime_status()

    assert status["journal_mode"].lower() == "wal"
    assert status["foreign_keys"] is True
    assert status["busy_timeout_ms"] > 0


def test_sqlite_runtime_tolerates_invalid_timeout(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_SQLITE_BUSY_TIMEOUT_MS", "not-a-number")
    assert db_runtime._configured_busy_timeout_ms() == 5_000


def test_sqlite_runtime_connection_is_usable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    connection: sqlite3.Connection = store._connect()
    try:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()
