"""Keep connector API user scopes bound to the authenticated principal.

A connector route reads the user it acts for from its own request body. This middleware
runs before routing, so it has to check every user the request names - in the query string
and in the body - against the authenticated principal, and refuse the request if any of
them is somebody else.

Checking only one of the two is the mistake this module exists to prevent: a request can
name the caller in the query string, where nothing reads it, and a different user in the
body, where the handler does.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import HTTPException, Request

from app.api.security.dependencies import enforce_user_match

logger = logging.getLogger("indoone.connector_security")

# Every mounted path whose request models carry a user_id in the body. A router added
# later with a user_id of its own must be added here too, or it will accept a user_id it
# does not check - which is exactly what happened to the YouTube routes.
#
# The connector routers for Canva and Facebook do not appear separately: their prefixes
_CONNECTOR_PATH_PREFIXES = (
    "/api/integrations",
    "/api/google-photos",
    # app/api/capabilities.py mounts at the root, so its user-scoped routes are listed one
    # by one.
    "/api/capabilities",
    "/api/memory",
    "/api/projects",
    "/api/tasks",
    "/api/agent",
)


def _is_connector_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in _CONNECTOR_PATH_PREFIXES)


def _named_users(request: Request, payload: Any) -> list[str]:
    """Every user this request names, from the query string and from the body.

    Both are returned rather than the first one found. The query string is not a
    substitute for the body: a handler reads the body, so a request that names the caller
    in the query and somebody else in the body must not pass on the strength of the query
    value alone.

    Only the top-level ``user_id`` of the body counts, because that is the field the
    request models declare and the handlers pass on. A user_id nested deeper is ordinary
    data - an email recipient, for instance - and is not the scope of the request.
    """
    named: list[str] = []

    query_user = request.query_params.get("user_id", "").strip()
    if query_user:
        named.append(query_user)

    if isinstance(payload, dict):
        body_user = payload.get("user_id")
        if isinstance(body_user, str) and body_user.strip():
            named.append(body_user.strip())

    return named


async def enforce_connector_user_scope(request: Request) -> None:
    """Keep connector API user scopes bound to the authenticated principal."""
    if not _is_connector_path(request.url.path):
        return

    payload: Any = None
    if request.method in {"POST", "PUT", "PATCH"}:
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
            except (TypeError, ValueError):
                payload = None

    named_users = _named_users(request, payload)
    if not named_users:
        return

    principal = str(getattr(request.state, "principal_id", "")).strip()
    if not principal:
        if os.getenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "false").strip().lower() == "true":
            raise HTTPException(status_code=401, detail="authenticated connector user required")
        logger.warning(
            "connector request without authenticated principal user_id=%s path=%s",
            named_users[0],
            request.url.path,
        )
        return

    # Check all of them, not just the first: any name that is not the caller is a refusal.
    for requested_user in named_users:
        if requested_user == principal:
            continue
        logger.warning(
            "connector user-scope mismatch principal=%s requested=%s path=%s",
            principal,
            requested_user,
            request.url.path,
        )
        enforce_user_match(request, requested_user)
