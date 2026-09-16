from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import HTTPException, Request

from app.api.dependencies import enforce_user_match

logger = logging.getLogger("indoone.connector_security")

_CONNECTOR_PATH_PREFIXES = ("/api/integrations", "/api/google-photos")


def _extract_requested_user(request: Request, payload: Any) -> str:
    query_user = request.query_params.get("user_id", "").strip()
    if query_user:
        return query_user
    if isinstance(payload, dict):
        value = payload.get("user_id")
        return value.strip() if isinstance(value, str) else ""
    return ""


async def enforce_connector_user_scope(request: Request) -> None:
    """Keep connector API user scopes bound to the authenticated principal."""
    if not request.url.path.startswith(_CONNECTOR_PATH_PREFIXES):
        return
    payload: Any = None
    if request.method in {"POST", "PUT", "PATCH"}:
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
            except (TypeError, ValueError):
                payload = None
    requested_user = _extract_requested_user(request, payload)
    if not requested_user:
        return

    principal = str(getattr(request.state, "principal_id", "")).strip()
    if not principal:
        if os.getenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false").strip().lower() == "true":
            raise HTTPException(status_code=401, detail="authenticated connector user required")
        logger.warning("connector request without authenticated principal user_id=%s path=%s", requested_user, request.url.path)
        return

    if requested_user != principal:
        logger.warning(
            "connector user-scope mismatch principal=%s requested=%s path=%s",
            principal,
            requested_user,
            request.url.path,
        )
        enforce_user_match(request, requested_user)
