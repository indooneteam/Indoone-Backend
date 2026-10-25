import sqlite3
from pathlib import Path

import pytest

from app.ai.memory.conversation_store import ConversationStore

OWNER = "user-one"
OTHER = "user-two"


def test_conversation_history_is_persistent_and_ordered(tmp_path: Path) -> None:
    db = tmp_path / "conversations.sqlite3"
    first = ConversationStore(db)
    first.append("c1", [("user", "Hello"), ("assistant", "Namaskara")], user_id=OWNER)

    second = ConversationStore(db)
    assert second.recent("c1", user_id=OWNER) == [("user", "Hello"), ("assistant", "Namaskara")]


def test_sqlite_uses_busy_timeout_and_wal_mode(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    with store._connect() as connection:
        busy_timeout = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    assert busy_timeout == 30_000
    assert journal_mode == "wal"


def test_sqlite_enforces_message_foreign_key_and_cascade(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.append("c1", [("user", "private")], user_id=OWNER)

    with store._connect() as connection:
        foreign_keys = connection.execute("PRAGMA foreign_key_list(messages)").fetchall()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                ("missing", "user", "orphan"),
            )
        connection.execute("DELETE FROM conversations WHERE conversation_id=?", ("c1",))
        remaining = int(
            connection.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", ("c1",)).fetchone()[0]
        )

    assert foreign_keys
    assert remaining == 0


def _legacy_database(path: Path, *, messages: list[tuple[str, str, str]]) -> None:
    """A database as an older release left it: no version table, no foreign key.

    Written with sqlite3 directly, not through the store, because the point is to
    arrive at the store the way an existing deployment does - a database that predates
    migrations, with data in it.
    """

    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT,
                status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                closed_at TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        for conversation_id, role, content in messages:
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
        connection.commit()
    finally:
        connection.close()


def test_a_legacy_database_gains_the_message_foreign_key(tmp_path: Path) -> None:
    """The repair is revision 0002 now, and it must still happen on an old database."""

    db = tmp_path / "conversations.sqlite3"
    _legacy_database(db, messages=[("c1", "user", "legacy")])
    connection = sqlite3.connect(str(db))
    try:
        connection.execute(
            "INSERT INTO conversations(conversation_id, user_id) VALUES (?, ?)", ("c1", OWNER)
        )
        connection.commit()
    finally:
        connection.close()

    store = ConversationStore(db)

    with store._connect() as connection:
        assert connection.execute("PRAGMA foreign_key_list(messages)").fetchall()
        assert (
            connection.execute(
                "SELECT content FROM messages WHERE conversation_id=?", ("c1",)
            ).fetchone()[0]
            == "legacy"
        )


def test_legacy_orphaned_messages_block_the_migration(tmp_path: Path) -> None:
    """A row with no conversation aborts the repair rather than being dropped."""

    db = tmp_path / "conversations.sqlite3"
    _legacy_database(db, messages=[("missing", "user", "orphan")])

    with pytest.raises(sqlite3.IntegrityError, match="orphaned messages"):
        ConversationStore(db)


def test_recent_history_is_bounded(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_messages=2)
    store.append(
        "c1",
        [("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four")],
        user_id=OWNER,
    )
    assert store.recent("c1", user_id=OWNER) == [("user", "three"), ("assistant", "four")]


def test_default_history_has_no_message_cap(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    rows = [(("user" if index % 2 == 0 else "assistant"), f"message-{index}") for index in range(60)]
    store.append("c1", rows, user_id=OWNER)

    assert store.message_count("c1", user_id=OWNER) == 60
    assert store.is_closed("c1", user_id=OWNER) is False
    assert store.recent("c1", user_id=OWNER) == rows


def test_conversation_closes_at_message_limit(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_messages=4)
    store.append("c1", [("user", "one"), ("assistant", "two"), ("user", "three")], user_id=OWNER)
    assert store.is_closed("c1", user_id=OWNER) is False
    store.append("c1", [("assistant", "four")], user_id=OWNER)
    assert store.message_count("c1", user_id=OWNER) == 4
    assert store.is_closed("c1", user_id=OWNER) is True
    with pytest.raises(ValueError, match="closed"):
        store.append("c1", [("user", "five")], user_id=OWNER)


def test_invalid_messages_are_rejected(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    with pytest.raises(ValueError, match="role"):
        store.append("c1", [("system", "bad")], user_id=OWNER)
    with pytest.raises(ValueError, match="content"):
        store.append("c1", [("user", "   ")], user_id=OWNER)


def test_cross_user_access_is_rejected(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.append("c1", [("user", "private")], user_id=OWNER)
    with pytest.raises(PermissionError, match="another user"):
        store.recent("c1", user_id=OTHER)


def test_ownerless_legacy_record_is_rejected(tmp_path: Path) -> None:
    db = tmp_path / "conversations.sqlite3"
    store = ConversationStore(db)
    with store._connect() as connection:
        connection.execute("INSERT INTO conversations(conversation_id, user_id) VALUES (?, NULL)", ("legacy",))
        connection.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, 'user', 'private')", ("legacy",))

    with pytest.raises(PermissionError, match="owner metadata"):
        store.recent("legacy", user_id=OWNER)
