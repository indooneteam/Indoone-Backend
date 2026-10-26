"""A thread can be named, and a thread can be shared.

Two small pieces of schema for the export and management suite:

* ``conversations.title`` - a user-chosen name. Until now a thread was identified by its
  id and whatever the first message happened to say, which is not something a person can
  find in a list.
* ``conversation_shares`` - one row per share link. The share reads the thread live rather
  than copying it: a copy would silently drift from the conversation the person believes
  they shared, and the owner can always stop sharing by deleting the row.

Both steps are conditional, so this revision is safe on a database the baseline created and
on one that has already been migrated.

Revision ID: 0003_title_and_shares
Revises: 0002_legacy_schema_repairs
Create Date: 2026-10-26
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "0003_title_and_shares"
down_revision = "0002_legacy_schema_repairs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = inspect(connection)

    columns = {column["name"] for column in inspector.get_columns("conversations")}

    if "title" not in columns:
        op.execute("ALTER TABLE conversations ADD COLUMN title TEXT")

    tables = set(inspector.get_table_names())

    if "conversation_shares" not in tables:
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_shares (
            share_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            user_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )
            """
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_shares_thread "
            "ON conversation_shares(conversation_id)"
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = inspect(connection)

    if "conversation_shares" in set(inspector.get_table_names()):
        op.execute("DROP TABLE IF EXISTS conversation_shares")

    columns = {column["name"] for column in inspector.get_columns("conversations")}

    if "title" in columns:
        op.execute("ALTER TABLE conversations DROP COLUMN title")
