"""Conversation history, on whichever database INDOONE_DATABASE_URL names.

SQLite is the default and its file is unchanged. The store talks to the database
through ``app.storage.database``, so the same code runs on Postgres, and the three
places that were written against SQLite specifically - the pragmas, the auto
incrementing key and the schema introspection - now ask the seam instead.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from app.storage import database
from app.storage.migrations import ensure_schema

DEFAULT_DB_PATH = Path(os.getenv("INDOONE_CONVERSATION_DB", "data/indoone_conversations.sqlite3"))
DEFAULT_MAX_MESSAGES: int | None = None


class ConversationStore:
    """Persistent conversation history with strict user ownership."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, max_messages: int | None = DEFAULT_MAX_MESSAGES) -> None:
        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages must be greater than zero when provided")
        self.db_path = db_path
        self.max_messages = max_messages
        self._initialize()

    def _connect(self) -> database.Connection:
        """Open the conversation store.

        The configured database URL wins; otherwise the file this store has always
        used is opened.
        """

        if database.is_postgres():
            return database.connect()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return database.connect(self.db_path)

    @staticmethod
    def _validate_integrity(connection: database.Connection) -> None:
        if connection.integrity_problems():
            raise database.IntegrityError("database foreign key integrity check failed")

    @staticmethod
    def _validate_integrity(connection: database.Connection) -> None:
        if connection.integrity_problems():
            raise database.IntegrityError("database foreign key integrity check failed")

    def _initialize(self) -> None:
        """Bring the schema up to date, then check it is sound.

        The two tables this store owns - ``conversations`` and ``messages`` - are
        created by the migrations now. The column that used to be added here if it
        was missing, and the table that used to be rebuilt here to gain its foreign
        key, are migrations too: schema surgery belongs in a revision with a version
        number, not in a constructor that runs on every boot.
        """

        with self._connect() as connection:
            if not database.is_postgres():
                # Write-ahead logging and a relaxed sync are SQLite settings. Postgres
                # has its own durability model and needs neither.
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
        ensure_schema(database.database_url(self.db_path))
        with self._connect() as connection:
            self._validate_integrity(connection)

    @staticmethod
    def _normalize_user_id(user_id: str | None) -> str:
        return (user_id or "").strip()

    @classmethod
    def _require_user_id(cls, user_id: str | None) -> str:
        normalized = cls._normalize_user_id(user_id)
        if not normalized or len(normalized) > 256:
            raise PermissionError("authenticated user required")
        return normalized

    @classmethod
    def _assert_owner(cls, row: Any | None, user_id: str | None) -> str:
        requested = cls._require_user_id(user_id)
        if row is None:
            raise ValueError("conversation not found")
        owner = str(row["user_id"] or "").strip()
        if not owner:
            raise PermissionError("conversation owner metadata is missing")
        if owner != requested:
            raise PermissionError("conversation belongs to another user")
        return requested

    def append(self, conversation_id: str, messages: Iterable[tuple[str, str]], user_id: str | None = None) -> None:
        rows = list(messages)
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        normalized_user_id = self._require_user_id(user_id)
        if not rows:
            return
        for role, content in rows:
            if role not in {"user", "assistant"}:
                raise ValueError("message role must be user or assistant")
            if not content.strip():
                raise ValueError("message content cannot be empty")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(conversation_id, user_id) VALUES (?, ?) "
                "ON CONFLICT(conversation_id) DO NOTHING",
                (conversation_id, normalized_user_id),
            )
            current = connection.execute("SELECT user_id, status FROM conversations WHERE conversation_id = ?", (conversation_id,)).fetchone()
            self._assert_owner(current, normalized_user_id)
            if current["status"] == "closed":
                raise ValueError("conversation is closed")
            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, role, content.strip()) for role, content in rows],
            )
            connection.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?", (conversation_id,))
            if self.max_messages is not None:
                count_row = connection.execute(
                    "SELECT COUNT(*) AS message_count FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                count = int(count_row["message_count"] if count_row is not None else 0)
                if count >= self.max_messages:
                    connection.execute("UPDATE conversations SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE conversation_id=?", (conversation_id,))

    def recent(self, conversation_id: str, user_id: str | None = None) -> list[tuple[str, str]]:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            owner = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(owner, user_id)
            if self.max_messages is None:
                rows = connection.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                    (conversation_id, self.max_messages),
                ).fetchall()
        if self.max_messages is None:
            return [(str(row["role"]), str(row["content"])) for row in rows]
        return [(str(row["role"]), str(row["content"])) for row in reversed(rows)]

    def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        owner = self._require_user_id(user_id)
        if limit <= 0 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT conversation_id, status, created_at, updated_at, closed_at FROM conversations "
                "WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def message_count(self, conversation_id: str, user_id: str | None = None) -> int:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        with self._connect() as connection:
            owner = connection.execute("SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            self._assert_owner(owner, user_id)
            row = connection.execute(
                "SELECT COUNT(*) AS message_count FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["message_count"] if row is not None else 0)

    def is_closed(self, conversation_id: str, user_id: str | None = None) -> bool:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
        owner_id = self._require_user_id(user_id)
        with self._connect() as connection:
            row = connection.execute("SELECT user_id, status FROM conversations WHERE conversation_id = ?", (conversation_id,)).fetchone()
            if row is None:
                return False
            self._assert_owner(row, owner_id)
        return str(row["status"]) == "closed"

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

    def rename(self, conversation_id: str, title: str, user_id: str | None = None) -> str:
        """Give a thread a name the user chose, and return it as stored.

        Cleaned before it is stored: a title is shown in a list, and a newline or a run of
        spaces in it makes that list look broken.
        """
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")

        cleaned = " ".join(str(title or "").split())

        if not cleaned:
            raise ValueError("title cannot be empty")

        if len(cleaned) > 120:
            raise ValueError("title must be at most 120 characters")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            self._assert_owner(row, user_id)
            connection.execute(
                "UPDATE conversations SET title=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE conversation_id=?",
                (cleaned, conversation_id),
            )

        return cleaned

    def create_share(self, conversation_id: str, user_id: str | None = None) -> str:
        """A non-guessable link id for this thread, owned by the person who made it."""
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")

        share_id = uuid4().hex

        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            self._assert_owner(row, user_id)
            connection.execute(
                "INSERT INTO conversation_shares (share_id, conversation_id, user_id) "
                "VALUES (?, ?, ?)",
                (share_id, conversation_id, user_id),
            )

        return share_id

    def shared_thread(self, share_id: str) -> list[tuple[str, str]] | None:
        """The messages behind a share link, or None when the link does not exist.

        No owner check on purpose: the whole point of a share link is that someone else can
        read it. It is read-only by construction - nothing here writes.
        """
        if not share_id:
            return None

        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id FROM conversation_shares WHERE share_id=?",
                (share_id,),
            ).fetchone()

            if row is None:
                return None

            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
                (str(row["conversation_id"]),),
            ).fetchall()

        return [(str(item["role"]), str(item["content"])) for item in rows]

    def purge_expired_closed(self, max_age_days: int = 7) -> int:
        """Delete conversations closed at least max_age_days ago.

        The cutoff is computed here rather than by a database function: both databases
        write created_at and closed_at as text starting YYYY-MM-DD HH:MM:SS, so a
        cutoff in that same shape compares correctly on either.
        """

        if max_age_days <= 0:
            raise ValueError("max_age_days must be greater than zero")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT conversation_id
                   FROM conversations
                  WHERE status='closed'
                    AND closed_at IS NOT NULL
                    AND closed_at <= ?""",
                (cutoff,),
            ).fetchall()
            if not rows:
                return 0
            ids = [str(row["conversation_id"]) for row in rows]
            connection.executemany(
                "DELETE FROM conversations WHERE conversation_id = ?",
                [(conversation_id,) for conversation_id in ids],
            )
            return len(ids)
