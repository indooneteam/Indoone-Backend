from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _db_path() -> Path:
    return Path(os.getenv("INDOONE_CAPABILITY_DB", "data/indoone_capabilities.db"))


# Reply rules are set by the business that connected a channel account. The
# limits match the ones the channel automation module already enforces.
_CHANNEL_RULE_TRIGGERS = frozenset({"comment", "message"})
_CHANNEL_RULE_MAX_PER_CHANNEL = 50
_CHANNEL_RULE_MAX_KEYWORD = 2200
_CHANNEL_RULE_MAX_RESPONSE = 2200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    with closing(_connect()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_user_key ON memories(user_id, key);
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user_updated ON memories(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                instructions TEXT NOT NULL,
                context TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                schedule TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
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
            );
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                integration_id TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_oauth_states_created ON oauth_states(created_at);
            CREATE TABLE IF NOT EXISTS oauth_state_results (
                state_hash TEXT NOT NULL,
                integration_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (state_hash, integration_id)
            );
            CREATE INDEX IF NOT EXISTS idx_oauth_state_results_created ON oauth_state_results(created_at);
            CREATE TABLE IF NOT EXISTS integration_tokens (
                user_id TEXT NOT NULL,
                integration_id TEXT NOT NULL,
                access_token BLOB NOT NULL,
                refresh_token BLOB,
                token_type TEXT NOT NULL,
                scope TEXT NOT NULL,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, integration_id)
            );
            CREATE TABLE IF NOT EXISTS pending_email_actions (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                integration_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                encrypted_payload BLOB NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (user_id, conversation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_pending_email_actions_expiry
                ON pending_email_actions(expires_at);
            CREATE TABLE IF NOT EXISTS channel_inbound_events (
                event_key TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('processing', 'sent', 'failed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_channel_inbound_events_updated
                ON channel_inbound_events(updated_at);
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
            );
            CREATE INDEX IF NOT EXISTS idx_channel_rules_user_channel
                ON channel_rules(user_id, channel);
            CREATE INDEX IF NOT EXISTS idx_integration_tokens_user ON integration_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
            CREATE INDEX IF NOT EXISTS idx_projects_user_updated ON projects(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_user_updated ON agent_runs(user_id, updated_at DESC);
            """
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(projects)").fetchall()}
        if "archived" not in columns:
            db.execute("ALTER TABLE projects ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        db.commit()


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def create_oauth_state(state: str, user_id: str, integration_id: str, redirect_uri: str) -> None:
    initialize()
    with closing(_connect()) as db:
        db.execute(
            "INSERT OR REPLACE INTO oauth_states(state, user_id, integration_id, redirect_uri, created_at) VALUES (?, ?, ?, ?, ?)",
            (state, user_id, integration_id, redirect_uri, _now()),
        )
        db.commit()


def consume_oauth_state(state: str, integration_id: str, max_age_seconds: int = 1800) -> dict[str, str] | None:
    """Consume one OAuth state under a write lock.

    This used to select the row and then delete it, with nothing holding the two
    statements together. Two callbacks carrying the same state could both read it and both
    go on to exchange the code. BEGIN IMMEDIATE takes the write lock before the read, so
    the second caller finds the row already gone.

    It also expires stale rows here rather than leaving that to the caller.

    app/capabilities/db_runtime.py used to install this behaviour by rebinding the name on
    the module object, which never took effect: every caller had already imported the
    function by name. The behaviour now lives in the function itself, where the callers
    actually look.
    """
    initialize()
    max_age_seconds = max(0, int(max_age_seconds))
    with closing(_connect()) as db:
        db.execute("BEGIN IMMEDIATE")
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        db.execute("DELETE FROM oauth_states WHERE created_at < ?", (cutoff,))
        row = db.execute(
            "SELECT user_id, integration_id, redirect_uri, created_at FROM oauth_states WHERE state = ? AND integration_id = ?",
            (state, integration_id),
        ).fetchone()
        if row is None:
            db.commit()
            return None
        db.execute("DELETE FROM oauth_states WHERE state = ? AND integration_id = ?", (state, integration_id))
        db.commit()
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
    except (TypeError, ValueError):
        return None
    age = (datetime.now(timezone.utc) - created).total_seconds()
    if age > max_age_seconds:
        return None
    return {"user_id": str(row["user_id"]), "integration_id": str(row["integration_id"]), "redirect_uri": str(row["redirect_uri"])}


def record_oauth_state_success(state: str, integration_id: str, user_id: str, provider_code: str) -> None:
    """Store a short-lived hash-only OAuth completion receipt for safe browser retries."""
    normalized_state = state.strip()
    normalized_integration = integration_id.strip().lower()
    normalized_user = user_id.strip()
    normalized_code = provider_code.strip()
    if not normalized_state or not normalized_integration or not normalized_user or not normalized_code:
        raise ValueError("OAuth completion receipt fields are required")
    initialize()
    state_hash = hashlib.sha256(normalized_state.encode("utf-8")).hexdigest()
    code_hash = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    with closing(_connect()) as db:
        cutoff = now.timestamp() - 24 * 60 * 60
        db.execute(
            "DELETE FROM oauth_state_results WHERE created_at < ?",
            (datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),),
        )
        db.execute(
            """
            INSERT OR REPLACE INTO oauth_state_results
                (state_hash, integration_id, user_id, code_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (state_hash, normalized_integration, normalized_user, code_hash, now.isoformat()),
        )
        db.commit()


def get_completed_oauth_state(
    state: str, integration_id: str, provider_code: str, max_age_seconds: int = 1800
) -> dict[str, str] | None:
    """Recognize only the exact same state+code after a successful exchange."""
    normalized_state = state.strip()
    normalized_integration = integration_id.strip().lower()
    normalized_code = provider_code.strip()
    if not normalized_state or not normalized_integration or not normalized_code:
        return None
    state_hash = hashlib.sha256(normalized_state.encode("utf-8")).hexdigest()
    code_hash = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, code_hash, created_at FROM oauth_state_results WHERE state_hash = ? AND integration_id = ?",
            (state_hash, normalized_integration),
        ).fetchone()
    if row is None or not hmac.compare_digest(str(row["code_hash"]), code_hash):
        return None
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
        age = (datetime.now(timezone.utc) - created).total_seconds()
    except (TypeError, ValueError):
        return None
    if age < 0 or age > max_age_seconds:
        return None
    return {"user_id": str(row["user_id"]), "integration_id": normalized_integration}


def get_completed_oauth_state_any_code(
    state: str, integration_id: str, max_age_seconds: int = 1800
) -> dict[str, str] | None:
    """Recognize a completed exchange for this state, whatever code was sent.

    The browser can resend the callback URL after the exchange has already
    succeeded (for example when the app completed it first), so the exact-code
    receipt may not match. The outcome is still "connected", so the page should
    report success instead of a failure.
    """
    normalized_state = state.strip()
    normalized_integration = integration_id.strip().lower()
    if not normalized_state or not normalized_integration:
        return None
    state_hash = hashlib.sha256(normalized_state.encode("utf-8")).hexdigest()
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, created_at FROM oauth_state_results WHERE state_hash = ? AND integration_id = ?",
            (state_hash, normalized_integration),
        ).fetchone()
    if row is None:
        return None
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
        age = (datetime.now(timezone.utc) - created).total_seconds()
    except (TypeError, ValueError):
        return None
    if age < 0 or age > max_age_seconds:
        return None
    return {"user_id": str(row["user_id"]), "integration_id": normalized_integration}


def upsert_integration_token(
    user_id: str,
    integration_id: str,
    access_token: bytes,
    refresh_token: bytes | None,
    token_type: str,
    scope: str,
    expires_at: str | None,
) -> None:
    initialize()
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute(
            """
            INSERT INTO integration_tokens(user_id, integration_id, access_token, refresh_token, token_type, scope, expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, integration_id) DO UPDATE SET
                access_token=excluded.access_token,
                refresh_token=COALESCE(excluded.refresh_token, integration_tokens.refresh_token),
                token_type=excluded.token_type,
                scope=excluded.scope,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at
            """,
            (user_id, integration_id, access_token, refresh_token, token_type, scope, expires_at, timestamp, timestamp),
        )
        db.commit()


def get_single_integration_token_user_id(integration_id: str) -> str | None:
    """Return the owner of the sole connected account for an integration.

    Webhook handlers do not carry the Indoone app's Firebase UID. If exactly one
    account is connected, its stored owner can be resolved without another
    environment variable. Ambiguous integrations deliberately fail closed.
    """
    normalized = integration_id.strip()
    if not normalized:
        return None
    initialize()
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT user_id FROM integration_tokens WHERE integration_id = ? ORDER BY updated_at DESC LIMIT 2",
            (normalized,),
        ).fetchall()
    if len(rows) != 1:
        return None
    user_id = str(rows[0]["user_id"] or "").strip()
    return user_id or None


def claim_channel_inbound_event(event_key: str, retry_after_seconds: int = 600) -> bool:
    normalized = event_key.strip()
    if not normalized or len(normalized) > 512:
        raise ValueError("event_key is required and must be at most 512 characters")
    initialize()
    timestamp = _now()
    with closing(_connect()) as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO channel_inbound_events(event_key, status, created_at, updated_at) VALUES (?, 'processing', ?, ?)",
            (normalized, timestamp, timestamp),
        )
        if cursor.rowcount > 0:
            db.commit()
            return True
        row = db.execute(
            "SELECT status, updated_at FROM channel_inbound_events WHERE event_key = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            return False
        status = str(row["status"])
        if status == "sent":
            return False
        if status == "processing":
            try:
                updated = datetime.fromisoformat(str(row["updated_at"]))
                age = (datetime.now(timezone.utc) - updated).total_seconds()
            except ValueError:
                age = 0
            if age < max(1, retry_after_seconds):
                return False
        cursor = db.execute(
            "UPDATE channel_inbound_events SET status='processing', updated_at=? WHERE event_key=?",
            (timestamp, normalized),
        )
        db.commit()
        return cursor.rowcount > 0


def complete_channel_inbound_event(event_key: str, success: bool) -> None:
    normalized = event_key.strip()
    if not normalized:
        return
    initialize()
    with closing(_connect()) as db:
        db.execute(
            "UPDATE channel_inbound_events SET status=?, updated_at=? WHERE event_key=?",
            ("sent" if success else "failed", _now(), normalized),
        )
        db.commit()


def get_integration_token(user_id: str, integration_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, integration_id, access_token, refresh_token, token_type, scope, expires_at, updated_at FROM integration_tokens WHERE user_id = ? AND integration_id = ?",
            (user_id, integration_id),
        ).fetchone()
    return _row_dict(row)


def get_integration_token_metadata(user_id: str, integration_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, integration_id, token_type, scope, expires_at, created_at, updated_at FROM integration_tokens WHERE user_id = ? AND integration_id = ?",
            (user_id, integration_id),
        ).fetchone()
    return _row_dict(row)


def delete_integration_token(user_id: str, integration_id: str) -> bool:
    """Delete only the selected user's saved credentials for one integration."""
    normalized_user = user_id.strip()
    normalized_integration = integration_id.strip().lower()
    if not normalized_user or not normalized_integration:
        raise ValueError("user_id and integration_id are required")
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute(
            "DELETE FROM integration_tokens WHERE user_id = ? AND integration_id = ?",
            (normalized_user, normalized_integration),
        )
        db.commit()
    return cursor.rowcount > 0


def save_pending_email_action(
    user_id: str,
    conversation_id: str,
    integration_id: str,
    action_type: str,
    encrypted_payload: bytes,
    expires_at: str,
) -> None:
    """Store one encrypted, user-scoped email action awaiting confirmation."""
    values = (
        user_id.strip(),
        conversation_id.strip(),
        integration_id.strip().lower(),
        action_type.strip().lower(),
        bytes(encrypted_payload),
        _now(),
        expires_at.strip(),
    )
    if not all(values[index] for index in (0, 1, 2, 3, 6)) or not values[4]:
        raise ValueError("pending email action fields are required")
    initialize()
    with closing(_connect()) as db:
        db.execute(
            """
            INSERT INTO pending_email_actions(
                user_id, conversation_id, integration_id, action_type,
                encrypted_payload, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                integration_id=excluded.integration_id,
                action_type=excluded.action_type,
                encrypted_payload=excluded.encrypted_payload,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at
            """,
            values,
        )
        db.commit()


def get_pending_email_action(user_id: str, conversation_id: str) -> dict[str, Any] | None:
    normalized_user = user_id.strip()
    normalized_conversation = conversation_id.strip()
    if not normalized_user or not normalized_conversation:
        return None
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            """
            SELECT user_id, conversation_id, integration_id, action_type,
                   encrypted_payload, created_at, expires_at
            FROM pending_email_actions
            WHERE user_id = ? AND conversation_id = ?
            """,
            (normalized_user, normalized_conversation),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            expires_at = datetime.fromisoformat(str(item["expires_at"]).replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except ValueError:
            expires_at = datetime.min.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            db.execute(
                "DELETE FROM pending_email_actions WHERE user_id = ? AND conversation_id = ?",
                (normalized_user, normalized_conversation),
            )
            db.commit()
            return None
    return item


def delete_pending_email_action(user_id: str, conversation_id: str) -> bool:
    normalized_user = user_id.strip()
    normalized_conversation = conversation_id.strip()
    if not normalized_user or not normalized_conversation:
        return False
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute(
            "DELETE FROM pending_email_actions WHERE user_id = ? AND conversation_id = ?",
            (normalized_user, normalized_conversation),
        )
        db.commit()
    return cursor.rowcount > 0


def upsert_memory(user_id: str, key: str, value: str, confidence: float, source: str) -> dict[str, Any]:
    initialize()
    timestamp = _now()
    with closing(_connect()) as db:
        existing = db.execute(
            "SELECT id, created_at FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        memory_id = str(existing["id"]) if existing else str(uuid4())
        created_at = str(existing["created_at"]) if existing else timestamp
        db.execute(
            """
            INSERT INTO memories(id, user_id, key, value, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (memory_id, user_id, key, value, confidence, source, created_at, timestamp),
        )
        db.commit()
        row = db.execute("SELECT * FROM memories WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
    return _row_dict(row) or {}


def list_memories(user_id: str, key_prefix: str = "", source: str = "", limit: int = 100) -> list[dict[str, Any]]:
    initialize()
    limit = max(1, min(limit, 500))
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]
    if key_prefix:
        clauses.append("key LIKE ?")
        params.append(f"{key_prefix}%")
    if source:
        clauses.append("source = ?")
        params.append(source)
    params.append(limit)
    with closing(_connect()) as db:
        rows = db.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def search_memories(user_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
    initialize()
    normalized = " ".join(query.split())
    if not normalized:
        return []
    limit = max(1, min(limit, 100))
    pattern = f"%{normalized}%"
    with closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND (key LIKE ? OR value LIKE ? OR source LIKE ?)
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, pattern, pattern, pattern, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_memory(user_id: str, memory_id: str) -> bool:
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute("DELETE FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id))
        db.commit()
    return cursor.rowcount > 0


def create_project(user_id: str, name: str, instructions: str, context: dict[str, Any]) -> dict[str, Any]:
    initialize()
    project_id = str(uuid4())
    timestamp = _now()
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    with closing(_connect()) as db:
        db.execute(
            "INSERT INTO projects(id, user_id, name, instructions, context, created_at, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (project_id, user_id, name, instructions, payload, timestamp, timestamp),
        )
        db.commit()
    return get_project(user_id, project_id) or {}


def get_project(user_id: str, project_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT * FROM projects WHERE user_id = ? AND id = ?",
            (user_id, project_id),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["context"] = json.loads(data["context"])
    data["archived"] = bool(data["archived"])
    return data


def update_project(user_id: str, project_id: str, name: str | None = None, instructions: str | None = None, context: dict[str, Any] | None = None, archived: bool | None = None) -> dict[str, Any] | None:
    """Merge a partial update into a project.

    The read and the write happen on one connection inside one transaction. They used to
    be two connections: the current row was fetched, the values were merged from that
    snapshot, and a second connection wrote the result. Two overlapping partial updates
    therefore each merged from the same stale row and the later write silently discarded
    the other one's field.
    """
    initialize()
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM projects WHERE user_id = ? AND id = ?",
            (user_id, project_id),
        ).fetchone()
        if row is None:
            db.commit()
            return None
        next_name = row["name"] if name is None else name
        next_instructions = row["instructions"] if instructions is None else instructions
        next_context = json.loads(row["context"]) if context is None else context
        next_archived = bool(row["archived"]) if archived is None else archived
        db.execute(
            "UPDATE projects SET name = ?, instructions = ?, context = ?, archived = ?, updated_at = ? WHERE user_id = ? AND id = ?",
            (next_name, next_instructions, json.dumps(next_context, ensure_ascii=False, sort_keys=True), 1 if next_archived else 0, timestamp, user_id, project_id),
        )
        db.commit()
    return get_project(user_id, project_id)


def delete_project(user_id: str, project_id: str) -> bool:
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute("DELETE FROM projects WHERE user_id = ? AND id = ?", (user_id, project_id))
        db.commit()
    return cursor.rowcount > 0


def list_projects(user_id: str, include_archived: bool = False, limit: int = 100) -> list[dict[str, Any]]:
    initialize()
    limit = max(1, min(limit, 500))
    clause = "user_id = ?" if include_archived else "user_id = ? AND archived = 0"
    with closing(_connect()) as db:
        rows = db.execute(f"SELECT * FROM projects WHERE {clause} ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [{**{key: row[key] for key in row.keys() if key != "context"}, "context": json.loads(row["context"]), "archived": bool(row["archived"])} for row in rows]


def search_projects(user_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
    initialize()
    normalized = " ".join(query.split())
    if not normalized:
        return []
    limit = max(1, min(limit, 100))
    pattern = f"%{normalized}%"
    with closing(_connect()) as db:
        rows = db.execute("SELECT * FROM projects WHERE user_id = ? AND archived = 0 AND (name LIKE ? OR instructions LIKE ? OR context LIKE ?) ORDER BY updated_at DESC LIMIT ?", (user_id, pattern, pattern, pattern, limit)).fetchall()
    return [{**{key: row[key] for key in row.keys() if key != "context"}, "context": json.loads(row["context"]), "archived": bool(row["archived"])} for row in rows]


def create_task(user_id: str, title: str, prompt: str, schedule: str, enabled: bool = True) -> dict[str, Any]:
    initialize()
    task_id = str(uuid4())
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute("INSERT INTO tasks(id, user_id, title, prompt, schedule, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (task_id, user_id, title, prompt, schedule, 1 if enabled else 0, timestamp, timestamp))
        db.commit()
    return {"id": task_id, "user_id": user_id, "title": title, "prompt": prompt, "schedule": schedule, "enabled": enabled, "created_at": timestamp, "updated_at": timestamp}


def list_tasks(user_id: str) -> list[dict[str, Any]]:
    initialize()
    with closing(_connect()) as db:
        rows = db.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
    return [{**dict(row), "enabled": bool(row["enabled"])} for row in rows]


def create_agent_run(user_id: str, message: str) -> dict[str, Any]:
    initialize()
    run_id = str(uuid4())
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute("INSERT INTO agent_runs(id, user_id, message, status, steps, results, blocked_steps, retry_counts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (run_id, user_id, message, "running", "[]", "[]", "[]", "[]", timestamp, timestamp))
        db.commit()
    return get_agent_run(user_id, run_id) or {}


def update_agent_run(user_id: str, run_id: str, status: str, steps: list[dict[str, Any]], results: list[dict[str, Any]], blocked_steps: list[dict[str, Any]], retry_counts: list[int]) -> dict[str, Any] | None:
    initialize()
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute("UPDATE agent_runs SET status = ?, steps = ?, results = ?, blocked_steps = ?, retry_counts = ?, updated_at = ? WHERE user_id = ? AND id = ?", (status, json.dumps(steps, ensure_ascii=False, sort_keys=True), json.dumps(results, ensure_ascii=False, sort_keys=True), json.dumps(blocked_steps, ensure_ascii=False, sort_keys=True), json.dumps(retry_counts), timestamp, user_id, run_id))
        db.commit()
    return get_agent_run(user_id, run_id)


def get_agent_run(user_id: str, run_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM agent_runs WHERE user_id = ? AND id = ?", (user_id, run_id)).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("steps", "results", "blocked_steps", "retry_counts"):
        data[key] = json.loads(data[key])
    return data


def list_agent_runs(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    initialize()
    limit = max(1, min(limit, 100))
    with closing(_connect()) as db:
        rows = db.execute("SELECT * FROM agent_runs WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [{**{key: row[key] for key in row.keys() if key not in {"steps", "results", "blocked_steps", "retry_counts"}}, "steps": json.loads(row["steps"]), "results": json.loads(row["results"]), "blocked_steps": json.loads(row["blocked_steps"]), "retry_counts": json.loads(row["retry_counts"])} for row in rows]


def _channel_rule_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["enabled"] = bool(data["enabled"])
    return data


def upsert_channel_rule(
    user_id: str,
    channel: str,
    trigger: str,
    keyword: str,
    response: str,
    enabled: bool = True,
    rule_id: str | None = None,
) -> dict[str, Any]:
    """Create or replace one reply rule for a connected channel account.

    Passing a rule_id updates that rule in place and keeps its original
    created_at; passing none creates a new one.
    """
    normalized_user = user_id.strip()
    normalized_channel = channel.strip().lower()
    normalized_trigger = trigger.strip().lower()
    normalized_keyword = keyword.strip().lower()
    normalized_response = response.strip()

    if not normalized_user:
        raise ValueError("user_id is required")
    if not normalized_channel:
        raise ValueError("channel is required")
    if normalized_trigger not in _CHANNEL_RULE_TRIGGERS:
        raise ValueError(f"trigger must be one of: {', '.join(sorted(_CHANNEL_RULE_TRIGGERS))}")
    if not normalized_response:
        raise ValueError("response is required")
    if len(normalized_keyword) > _CHANNEL_RULE_MAX_KEYWORD:
        raise ValueError(f"keyword must be at most {_CHANNEL_RULE_MAX_KEYWORD} characters")
    if len(normalized_response) > _CHANNEL_RULE_MAX_RESPONSE:
        raise ValueError(f"response must be at most {_CHANNEL_RULE_MAX_RESPONSE} characters")

    initialize()
    timestamp = _now()
    identifier = (rule_id or "").strip()
    existing = get_channel_rule(normalized_user, identifier) if identifier else None
    if not identifier:
        identifier = uuid4().hex
    created_at = str(existing["created_at"]) if existing else timestamp

    with closing(_connect()) as db:
        if existing is None:
            stored = db.execute(
                "SELECT COUNT(*) FROM channel_rules WHERE user_id = ? AND channel = ?",
                (normalized_user, normalized_channel),
            ).fetchone()[0]
            if stored >= _CHANNEL_RULE_MAX_PER_CHANNEL:
                raise ValueError(f"a channel may hold at most {_CHANNEL_RULE_MAX_PER_CHANNEL} rules")
        db.execute(
            """
            INSERT INTO channel_rules(user_id, rule_id, channel, trigger, keyword, response, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, rule_id) DO UPDATE SET
                channel=excluded.channel,
                trigger=excluded.trigger,
                keyword=excluded.keyword,
                response=excluded.response,
                enabled=excluded.enabled,
                updated_at=excluded.updated_at
            """,
            (
                normalized_user,
                identifier,
                normalized_channel,
                normalized_trigger,
                normalized_keyword,
                normalized_response,
                1 if enabled else 0,
                created_at,
                timestamp,
            ),
        )
        db.commit()

    return get_channel_rule(normalized_user, identifier) or {}


def get_channel_rule(user_id: str, rule_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, rule_id, channel, trigger, keyword, response, enabled, created_at, updated_at FROM channel_rules WHERE user_id = ? AND rule_id = ?",
            (user_id.strip(), rule_id.strip()),
        ).fetchone()
    return _channel_rule_row(row)


def list_channel_rules(
    user_id: str,
    channel: str | None = None,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    initialize()
    query = "SELECT user_id, rule_id, channel, trigger, keyword, response, enabled, created_at, updated_at FROM channel_rules WHERE user_id = ?"
    parameters: list[Any] = [user_id.strip()]
    if channel:
        query += " AND channel = ?"
        parameters.append(channel.strip().lower())
    if enabled_only:
        query += " AND enabled = 1"
    query += " ORDER BY created_at ASC, rule_id ASC"

    with closing(_connect()) as db:
        rows = db.execute(query, tuple(parameters)).fetchall()
    return [rule for rule in (_channel_rule_row(row) for row in rows) if rule is not None]


def set_channel_rule_enabled(user_id: str, rule_id: str, enabled: bool) -> bool:
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute(
            "UPDATE channel_rules SET enabled = ?, updated_at = ? WHERE user_id = ? AND rule_id = ?",
            (1 if enabled else 0, _now(), user_id.strip(), rule_id.strip()),
        )
        db.commit()
    return cursor.rowcount > 0


def delete_channel_rule(user_id: str, rule_id: str) -> bool:
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute(
            "DELETE FROM channel_rules WHERE user_id = ? AND rule_id = ?",
            (user_id.strip(), rule_id.strip()),
        )
        db.commit()
    return cursor.rowcount > 0
