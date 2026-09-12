from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable


DEFAULT_DB_PATH = Path(os.getenv("INDOONE_CONVERSATION_DB", "data/indoone_conversations.sqlite3"))
DEFAULT_MAX_MESSAGES = 50


class ConversationStore:
    """Persistent SQLite conversation history with bounded lifecycle and optional user ownership."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        self.db_path = db_path
        self.max_messages = max_messages
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    closed_at TEXT
                )
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(conversations)").fetchall()}
            if "user_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id ON messages(conversation_id, id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_status_updated ON conversations(status, updated_at)"
            )

    @staticmethod
    def _normalize_user_id(user_id: str | None) -> str:
        return (user_id or "").strip()

    def _assert_owner(self, row: sqlite3.Row | None, user_id: str | None) -> None:
        requested = self._normalize_user_id(user_id)
        if row is None or not requested:
            return
        owner = str(row["user_id"] or "")
        if owner and owner != requested:
            raise PermissionError("conversation belongs to another user")

    def append(self, conversation_id: str, messages: Iterable[tuple[str, str]], user_id: str | None = None) -> None:
        rows = list(messages)
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        if not rows:
            return
        for role, content in rows:
            if role not in {"user", "assistant"}:
                raise ValueError("message role must be user or assistant")
            if not content.strip():
                raise ValueError("message content cannot be empty")

        normalized_user_id = self._normalize_user_id(user_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(conversation_id, user_id) VALUES (?, ?)",
                (conversation_id, normalized_user_id or None),
            )
            current = connection.execute(
                "SELECT user_id, status FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            self._assert_owner(current, normalized_user_id)
            if current is not None and current["status"] == "closed":
                raise ValueError("conversation is closed")
            if current is not None and not current["user_id"] and normalized_user_id:
                connection.execute("UPDATE conversations SET user_id=? WHERE conversation_id=?", (normalized_user_id, conversation_id))

            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, role, content.strip()) for role, content in rows],
            )
            connection.execute(
                "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                (conversation_id,),
            )

            count = int(connection.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchone()[0])
            if count >= self.max_messages:
                connection.execute(
                    "UPDATE conversations SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                    (conversation_id,),
                )

    def recent(self, conversation_id: str, user_id: str | None = None) -> list[tuple[str, str]]:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            owner = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(owner, user_id)
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, self.max_messages),
            ).fetchall()
        return [(str(row["role"]), str(row["content"])) for row in reversed(rows)]

    def message_count(self, conversation_id: str, user_id: str | None = None) -> int:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            owner = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(owner, user_id)
            row = connection.execute("SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchone()
        return int(row["count"] if row is not None else 0)

    def is_closed(self, conversation_id: str, user_id: str | None = None) -> bool:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            row = connection.execute("SELECT user_id, status FROM conversations WHERE conversation_id = ?", (conversation_id,)).fetchone()
            self._assert_owner(row, user_id)
        return row is not None and str(row["status"]) == "closed"

    def close(self, conversation_id: str, user_id: str | None = None) -> None:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            row = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(row, user_id)
            connection.execute("UPDATE conversations SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE conversation_id=?", (conversation_id,))

    def delete(self, conversation_id: str, user_id: str | None = None) -> None:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            row = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(row, user_id)
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
