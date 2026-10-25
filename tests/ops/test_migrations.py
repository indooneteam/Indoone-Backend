"""The schema is Alembic's, and these tests hold that line.

Four things must stay true, and each of them has broken a deployment somewhere before
migrations existed:

* a fresh database ends up with the whole schema and a recorded revision;
* a database that predates migrations is adopted without losing a row;
* running the upgrade twice is not an error;
* a service told not to migrate does not quietly create tables anyway.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.storage import database, migrations


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "schema.db"))
    monkeypatch.delenv("INDOONE_DATABASE_URL", raising=False)
    monkeypatch.delenv("INDOONE_MIGRATE_ON_START", raising=False)
    migrations.reset_for_tests()
    yield
    migrations.reset_for_tests()


def _tables() -> set[str]:
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {str(row[0]) for row in rows}


def test_a_fresh_database_gets_the_whole_schema_and_a_revision() -> None:
    assert migrations.current_revision() is None

    revision = migrations.ensure_schema()

    assert revision == migrations.BASELINE_REVISION or revision is not None
    assert migrations.current_revision() == revision
    # Every table the application reads, in one assertion: a missing one is a table
    # some store will query and not find.
    assert {
        "conversations",
        "messages",
        "memories",
        "projects",
        "tasks",
        "agent_runs",
        "oauth_states",
        "oauth_state_results",
        "integration_tokens",
        "pending_email_actions",
        "channel_inbound_events",
        "channel_rules",
        "notification_tokens",
        "usage_counters",
    } <= _tables()


def test_running_the_upgrade_twice_is_not_an_error() -> None:
    first = migrations.upgrade_to_head()
    second = migrations.upgrade_to_head()

    assert first == second


def test_ensure_schema_is_cheap_the_second_time() -> None:
    """It is called by every store's initialize(), so it must not redo the work."""

    migrations.ensure_schema()
    revision = migrations.current_revision()

    assert migrations.ensure_schema() == revision


def test_a_database_that_predates_migrations_is_adopted_with_its_data(
    tmp_path: Path, monkeypatch
) -> None:
    """The upgrade an existing deployment actually performs: no version table, real rows."""

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                closed_at TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO conversations(conversation_id, user_id) VALUES ('c1', 'u1')"
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(path))
    migrations.reset_for_tests()

    migrations.ensure_schema()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = 'c1'"
        ).fetchone()
    assert row is not None and row[0] == "u1"
    assert migrations.current_revision() is not None


def test_a_service_told_not_to_migrate_creates_nothing(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MIGRATE_ON_START", "0")

    revision = migrations.ensure_schema()

    assert revision is None
    assert "conversations" not in _tables()


def test_migrate_on_start_is_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_MIGRATE_ON_START", raising=False)

    assert migrations.migrate_on_start() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_migrate_on_start_can_be_switched_off(monkeypatch, value: str) -> None:
    monkeypatch.setenv("INDOONE_MIGRATE_ON_START", value)

    assert migrations.migrate_on_start() is False
