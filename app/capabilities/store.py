from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _db_path() -> Path:
    return Path(os.getenv("INDOONE_CAPABILITY_DB", "data/indoone_capabilities.db"))


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


def consume_oauth_state(state: str, integration_id: str, max_age_seconds: int = 600) -> dict[str, str] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT * FROM oauth_states WHERE state = ? AND integration_id = ?",
            (state, integration_id),
        ).fetchone()
        if row is None:
            return None
        db.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
        db.commit()
    created = datetime.fromisoformat(str(row["created_at"]))
    age = (datetime.now(timezone.utc) - created).total_seconds()
    if age > max_age_seconds:
        return None
    return {"user_id": str(row["user_id"]), "integration_id": str(row["integration_id"]), "redirect_uri": str(row["redirect_uri"])}


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
                refresh_token=excluded.refresh_token,
                token_type=excluded.token_type,
                scope=excluded.scope,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at
            """,
            (user_id, integration_id, access_token, refresh_token, token_type, scope, expires_at, timestamp, timestamp),
        )
        db.commit()


def get_integration_token(user_id: str, integration_id: str) -> dict[str, Any] | None:
    initialize()
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT user_id, integration_id, access_token, token_type, scope, expires_at, updated_at FROM integration_tokens WHERE user_id = ? AND integration_id = ?",
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
    initialize()
    current = get_project(user_id, project_id)
    if current is None:
        return None
    timestamp = _now()
    next_name = current["name"] if name is None else name
    next_instructions = current["instructions"] if instructions is None else instructions
    next_context = current["context"] if context is None else context
    next_archived = current["archived"] if archived is None else archived
    with closing(_connect()) as db:
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
