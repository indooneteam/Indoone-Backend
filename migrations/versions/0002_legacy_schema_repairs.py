"""Repairs that used to happen inside the stores on every boot.

Two pieces of schema surgery lived in application code before migrations existed, and
both are here instead.

* ``projects.archived`` was added by an ``ALTER TABLE`` in the capability store when
  the column was missing.
* ``messages`` was rebuilt by the conversation store when the table had no foreign key
  to ``conversations``, after refusing to do so if that would orphan any row.

Neither is a new feature. They are here because a schema change belongs in a revision
that runs once, records that it ran, and can be reviewed before it touches production -
not in a constructor that runs on every start.

Both steps are conditional, so this revision is safe on a database that the baseline
created fresh (where nothing needs doing) and on one that predates the baseline.

Everything runs on the migration's own connection (``op.get_bind()``), never on a
second one opened by the application: on SQLite a separate connection does not see
tables created inside the migration's open transaction.

Revision ID: 0002_legacy_schema_repairs
Revises: 0001_baseline
Create Date: 2026-10-25
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

from app.storage import database

revision = "0002_legacy_schema_repairs"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    _add_projects_archived(bind, inspector, tables)
    _add_messages_foreign_key(bind, inspector, tables)


def _add_projects_archived(bind, inspector, tables: set[str]) -> None:
    if "projects" not in tables:
        return
    if "archived" in {column["name"] for column in inspector.get_columns("projects")}:
        return
    bind.execute(
        text("ALTER TABLE projects ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    )


def _add_messages_foreign_key(bind, inspector, tables: set[str]) -> None:
    """Rebuild ``messages`` with its foreign key, if it does not already have one.

    The rebuild is the portable route: SQLite cannot add a constraint to an existing
    table, and a table copy that both databases support is one path instead of two.
    A row that would be orphaned aborts the migration - the same rule the store
    enforced, because dropping messages to satisfy a constraint is not a repair.
    """

    if "messages" not in tables:
        return
    if inspector.get_foreign_keys("messages"):
        return

    orphan_count = int(
        bind.execute(
            text(
                """SELECT COUNT(*) AS orphan_count
                     FROM messages AS messages
                     LEFT JOIN conversations AS conversations
                       ON conversations.conversation_id = messages.conversation_id
                    WHERE conversations.conversation_id IS NULL"""
            )
        ).scalar()
        or 0
    )
    if orphan_count:
        # The same exception and the same wording the store raised before this became a
        # migration: callers and tests already handle it, and "orphaned messages" is
        # what the operator needs to search for.
        raise database.IntegrityError(
            f"cannot enable message foreign key with orphaned messages: {orphan_count} "
            "rows have no conversation. Repair or delete them, then run the migration again."
        )

    identity = database.identity_column()
    bind.execute(text("DROP INDEX IF EXISTS idx_messages_conversation_id_id"))
    bind.execute(
        text(
            f"""CREATE TABLE IF NOT EXISTS messages_rebuilt (
                id {identity},
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            )"""
        )
    )
    bind.execute(
        text(
            """INSERT INTO messages_rebuilt(id, conversation_id, role, content, created_at)
               SELECT id, conversation_id, role, content, created_at
                 FROM messages"""
        )
    )
    bind.execute(text("DROP TABLE messages"))
    bind.execute(text("ALTER TABLE messages_rebuilt RENAME TO messages"))
    bind.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id "
            "ON messages(conversation_id, id)"
        )
    )


def downgrade() -> None:
    """No downgrade.

    The upgrade only adds a defaulted column and a constraint. Reversing it would mean
    dropping a column that holds data, or rebuilding a table to lose a constraint the
    application relies on. A downgrade that destroys data is worse than no downgrade.
    """

    pass
