from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

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


def test_sqlite_runtime_serializes_schema_initialization(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: store.initialize(), range(8)))

    with store._connect() as db:
        assert db.execute("SELECT 1").fetchone()[0] == 1


def test_oauth_state_is_single_use_and_bindings_are_preserved(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    store.initialize()
    store.create_oauth_state("state-1", "user-one", "google_drive", "https://example.test/callback")

    consumed = store.consume_oauth_state("state-1", "google_drive")
    replay = store.consume_oauth_state("state-1", "google_drive")

    assert consumed == {
        "user_id": "user-one",
        "integration_id": "google_drive",
        "redirect_uri": "https://example.test/callback",
    }
    assert replay is None


def test_oauth_state_wrong_integration_does_not_consume_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    store.initialize()
    store.create_oauth_state("state-2", "user-one", "google_drive", "https://example.test/callback")

    assert store.consume_oauth_state("state-2", "gmail") is None
    assert store.consume_oauth_state("state-2", "google_drive") is not None


def test_oauth_state_parallel_consumption_allows_only_one_winner(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    store.initialize()
    store.create_oauth_state("state-race", "user-one", "google_drive", "https://example.test/callback")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.consume_oauth_state("state-race", "google_drive"), range(2)))

    assert sum(result is not None for result in results) == 1


def test_oauth_state_cleanup_removes_expired_entries(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    db_runtime.configure_sqlite_runtime()
    store.initialize()
    store.create_oauth_state("expired", "user-one", "google_drive", "https://example.test/callback")
    store.create_oauth_state("fresh", "user-one", "google_drive", "https://example.test/callback")

    with store._connect() as db:
        db.execute("UPDATE oauth_states SET created_at = ? WHERE state = ?", ("2000-01-01T00:00:00+00:00", "expired"))
        db.commit()

    assert store.consume_oauth_state("fresh", "google_drive") is not None
    with store._connect() as db:
        assert db.execute("SELECT 1 FROM oauth_states WHERE state = ?", ("expired",)).fetchone() is None
