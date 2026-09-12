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
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, key)
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                instructions TEXT NOT NULL,
                context TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
            """
        )
        db.commit()


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def upsert_memory(user_id: str, key: str, value: str, confidence: float, source: str) -> dict[str, Any]:
    initialize()
    timestamp = _now()
    memory_id = str(uuid4())
    with closing(_connect()) as db:
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
            (memory_id, user_id, key, value, confidence, source, timestamp, timestamp),
        )
        db.commit()
        row = db.execute(
            "SELECT * FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
    return _row_dict(row) or {}


def list_memories(user_id: str) -> list[dict[str, Any]]:
    initialize()
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_memory(user_id: str, memory_id: str) -> bool:
    initialize()
    with closing(_connect()) as db:
        cursor = db.execute(
            "DELETE FROM memories WHERE user_id = ? AND id = ?",
            (user_id, memory_id),
        )
        db.commit()
    return cursor.rowcount > 0


def create_project(user_id: str, name: str, instructions: str, context: dict[str, Any]) -> dict[str, Any]:
    initialize()
    project_id = str(uuid4())
    timestamp = _now()
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    with closing(_connect()) as db:
        db.execute(
            "INSERT INTO projects(id, user_id, name, instructions, context, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, user_id, name, instructions, payload, timestamp, timestamp),
        )
        db.commit()
    return {
        "id": project_id,
        "user_id": user_id,
        "name": name,
        "instructions": instructions,
        "context": context,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def list_projects(user_id: str) -> list[dict[str, Any]]:
    initialize()
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [
        {
            **{key: row[key] for key in row.keys() if key != "context"},
            "context": json.loads(row["context"]),
        }
        for row in rows
    ]


def create_task(user_id: str, title: str, prompt: str, schedule: str, enabled: bool = True) -> dict[str, Any]:
    initialize()
    task_id = str(uuid4())
    timestamp = _now()
    with closing(_connect()) as db:
        db.execute(
            "INSERT INTO tasks(id, user_id, title, prompt, schedule, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, user_id, title, prompt, schedule, 1 if enabled else 0, timestamp, timestamp),
        )
        db.commit()
    return {
        "id": task_id,
        "user_id": user_id,
        "title": title,
        "prompt": prompt,
        "schedule": schedule,
        "enabled": enabled,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def list_tasks(user_id: str) -> list[dict[str, Any]]:
    initialize()
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [{**dict(row), "enabled": bool(row["enabled"])} for row in rows]
