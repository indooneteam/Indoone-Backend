from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable


DEFAULT_DB_PATH = Path(os.getenv("INDOONE_CONVERSATION_DB", "data/indoone_conversations.sqlite3"))


class ConversationStore:
    """Small persistent SQLite-backed conversation history store."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, max_messages: int = 40) -> None:
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
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id "
                "ON messages(conversation_id, id)"
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
            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, role, content.strip()) for role, content in rows],
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

    def delete(self, conversation_id: str) -> None:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
