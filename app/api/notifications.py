"""Push-notification token registration for the Indoone app.

The Android client registers its device push token here after sign-in
(see NotificationApi.kt). Tokens are stored per user so the backend can
address a device when it has something to deliver.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.dependencies import current_user_id

router = APIRouter(tags=["notifications"])

MAX_TOKEN_LENGTH = 4096
DEFAULT_PLATFORM = "android"


def _db_path() -> Path:
    return Path(os.getenv("INDOONE_NOTIFICATION_DB", "data/indoone_notifications.db"))


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    """Create the notification-token table if it does not exist yet."""
    with closing(_connect()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS notification_tokens (
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                platform TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, token)
            );
            CREATE INDEX IF NOT EXISTS idx_notification_tokens_user
                ON notification_tokens(user_id);
            """
        )
        db.commit()


def stored_tokens(user_id: str) -> list[str]:
    """Return the push tokens registered for a user, newest first."""
    with closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT token
              FROM notification_tokens
             WHERE user_id = ?
             ORDER BY updated_at DESC, token ASC
            """,
            (user_id.strip(),),
        ).fetchall()
    return [str(row["token"]) for row in rows]


class RegisterTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=MAX_TOKEN_LENGTH)
    platform: str = Field(default=DEFAULT_PLATFORM, max_length=32)


@router.post("/notifications/register")
async def register_notification_token(
    request: Request,
    body: RegisterTokenRequest,
) -> dict[str, object]:
    """Store the device push token for the signed-in user."""
    user_id = current_user_id(request)

    token = body.token.strip()
    if not token:
        raise HTTPException(status_code=422, detail="token is required")

    platform = body.platform.strip() or DEFAULT_PLATFORM
    updated_at = datetime.now(timezone.utc).isoformat()

    with closing(_connect()) as db:
        db.execute(
            """
            INSERT INTO notification_tokens (user_id, token, platform, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, token) DO UPDATE SET
                platform = excluded.platform,
                updated_at = excluded.updated_at
            """,
            (user_id, token, platform, updated_at),
        )
        db.commit()

    return {"ok": True, "registered": True, "platform": platform}
