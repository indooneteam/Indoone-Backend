"""The schema as it stood when migrations were introduced.

This baseline is the whole schema in one revision: every table and index that used to
be created inline by the store modules, in the order the foreign keys require. It uses
``IF NOT EXISTS`` on purpose, so a half-built database converges instead of failing;
an existing deployment is adopted by stamping this revision rather than running it.

Two column types are asked for rather than written: the auto-incrementing identity
column and the binary blob column have no syntax in common between SQLite and
Postgres, and ``app.storage.database`` is the one place that decides them.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-25
"""

from __future__ import annotations

from alembic import op

from app.storage import database

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

# Dropped in reverse dependency order, so the foreign key from messages to
# conversations never blocks a downgrade.
_TABLES_IN_DROP_ORDER = (
    "usage_counters",
    "notification_tokens",
    "channel_rules",
    "channel_inbound_events",
    "pending_email_actions",
    "integration_tokens",
    "oauth_state_results",
    "oauth_states",
    "agent_runs",
    "tasks",
    "projects",
    "memories",
    "messages",
    "conversations",
)


def upgrade() -> None:
    identity = database.identity_column()
    blob = database.blob_type()

    # conversations
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        user_id TEXT,
        status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        closed_at TEXT
        )
    """)

    # messages
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS messages (
        id {identity},
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        )
    """)

    # memories
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
        )
    """)

    # projects
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        instructions TEXT NOT NULL,
        context TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        archived INTEGER NOT NULL DEFAULT 0
        )
    """)

    # tasks
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        prompt TEXT NOT NULL,
        schedule TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
        )
    """)

    # agent_runs
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS agent_runs (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT NOT NULL,
        steps TEXT NOT NULL,
        results TEXT NOT NULL,
        blocked_steps TEXT NOT NULL,
        retry_counts TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
        )
    """)

    # oauth_states
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS oauth_states (
        state TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        integration_id TEXT NOT NULL,
        redirect_uri TEXT NOT NULL,
        created_at TEXT NOT NULL
        )
    """)

    # oauth_state_results
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS oauth_state_results (
        state_hash TEXT NOT NULL,
        integration_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (state_hash, integration_id)
        )
    """)

    # integration_tokens
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS integration_tokens (
        user_id TEXT NOT NULL,
        integration_id TEXT NOT NULL,
        access_token {blob} NOT NULL,
        refresh_token {blob},
        token_type TEXT NOT NULL,
        scope TEXT NOT NULL,
        expires_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, integration_id)
        )
    """)

    # pending_email_actions
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS pending_email_actions (
        user_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        integration_id TEXT NOT NULL,
        action_type TEXT NOT NULL,
        encrypted_payload {blob} NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        PRIMARY KEY (user_id, conversation_id)
        )
    """)

    # channel_inbound_events
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS channel_inbound_events (
        event_key TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK(status IN ('processing', 'sent', 'failed')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
        )
    """)

    # channel_rules
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS channel_rules (
        user_id TEXT NOT NULL,
        rule_id TEXT NOT NULL,
        channel TEXT NOT NULL,
        trigger TEXT NOT NULL CHECK(trigger IN ('comment', 'message')),
        keyword TEXT NOT NULL,
        response TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, rule_id)
        )
    """)

    # notification_tokens
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS notification_tokens (
        user_id TEXT NOT NULL,
        token TEXT NOT NULL,
        platform TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, token)
        )
    """)

    # usage_counters
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS usage_counters (
        subject TEXT NOT NULL,
        window_key TEXT NOT NULL,
        used INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (subject, window_key)
        )
    """)

    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_agent_runs_user_updated ON agent_runs(user_id, updated_at DESC)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_channel_inbound_events_updated
        ON channel_inbound_events(updated_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_channel_rules_user_channel
        ON channel_rules(user_id, channel)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_conversations_status_updated ON conversations(status, updated_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_integration_tokens_user ON integration_tokens(user_id)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memories_user_updated ON memories(user_id, updated_at DESC)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id ON messages(conversation_id, id)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_notification_tokens_user
        ON notification_tokens(user_id)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_oauth_state_results_created ON oauth_state_results(created_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_oauth_states_created ON oauth_states(created_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_pending_email_actions_expiry
        ON pending_email_actions(expires_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_projects_user_updated ON projects(user_id, updated_at DESC)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)
    """)
    op.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_user_key ON memories(user_id, key)
    """)


def downgrade() -> None:
    for table in _TABLES_IN_DROP_ORDER:
        op.execute(f"DROP TABLE IF EXISTS {table}")
