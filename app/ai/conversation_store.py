from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable


DEFAULT_DB_PATH = Path(os.getenv("INDOONE_CONVERSATION_DB", "data/indoone_conversations.sqlite3"))
DEFAULT_MAX_MESSAGES = 50


class ConversationStore:
    """Persistent SQLite conversation history with a bounded lifecycle."""

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
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    closed_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id "
                "ON messages(conversation_id, id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_status_updated "
                "ON conversations(status, updated_at)"
            )

    def append(self, conversation_id: str, messages: Iterable[tuple[str, str]]) -> None:
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

        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(conversation_id) VALUES (?)",
                (conversation_id,),
            )
            current = connection.execute(
                "SELECT status FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if current is not None and current["status"] == "closed":
                raise ValueError("conversation is closed")

            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, role, content.strip()) for role, content in rows],
            )
            connection.execute(
                "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                (conversation_id,),
            )

            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            if count >= self.max_messages:
                connection.execute(
                    "UPDATE conversations SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                    (conversation_id,),
                )

    def recent(self, conversation_id: str) -> list[tuple[str, str]]:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, self.max_messages),
            ).fetchall()
        return [(str(row["role"]), str(row["content"])) for row in reversed(rows)]

    def message_count(self, conversation_id: str) -> int:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def is_closed(self, conversation_id: str) -> bool:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None and str(row["status"]) == "closed"

    def close(self, conversation_id: str) -> None:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                (conversation_id,),
            )

    def delete(self, conversation_id: str) -> None:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
