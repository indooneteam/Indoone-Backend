import sqlite3
from pathlib import Path

import pytest

from app.ai.conversation_store import ConversationStore

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


def test_legacy_messages_schema_is_migrated_to_foreign_key(tmp_path: Path) -> None:
    db = tmp_path / "conversations.sqlite3"
    store = ConversationStore(db)
    with store._connect() as connection:
        connection.execute("INSERT INTO conversations(conversation_id, user_id) VALUES (?, ?)", ("c1", OWNER))
        connection.execute("DROP TABLE messages")
        connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        connection.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)", ("c1", "user", "legacy"))

    reopened = ConversationStore(db)
    with reopened._connect() as connection:
        assert connection.execute("PRAGMA foreign_key_list(messages)").fetchall()
        assert connection.execute("SELECT content FROM messages WHERE conversation_id=?", ("c1",)).fetchone()[0] == "legacy"


def test_legacy_orphaned_messages_block_migration(tmp_path: Path) -> None:
    db = tmp_path / "conversations.sqlite3"
    store = ConversationStore(db)
    with store._connect() as connection:
        connection.execute("DROP TABLE messages")
        connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        connection.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)", ("missing", "user", "orphan"))

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
