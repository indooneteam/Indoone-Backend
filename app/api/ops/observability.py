"""The operator's view of what the backend just did.

This is the read side of ``app/ai/observability.py``: the aggregate an operator looks
at to answer "what did that message cost, and where did the time go".

It is off unless an operator turns it on. Set ``INDOONE_OBSERVABILITY_TOKEN`` to a
long random string and call the endpoint with it in the
``X-Indoone-Observability-Token`` header. While the setting is empty the endpoint
answers 404, because a metrics endpoint that is exposed by default is a leak, and the
default should be the safe one.

The snapshot holds counts, names, durations and tokens only. It never holds a prompt,
a completion, a tool payload or a user id.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from app.ai.gateway.observability import reset, snapshot

router = APIRouter(tags=["observability"])

def _configured_token() -> str:
    return os.getenv("INDOONE_OBSERVABILITY_TOKEN", "").strip()


def _require_token(provided: str) -> None:
    configured = _configured_token()
    if not configured:
        # Not configured means not exposed.
        raise HTTPException(status_code=404, detail="not found")
    if provided.strip() != configured:
        raise HTTPException(status_code=401, detail="invalid observability token")


@router.get("/ops/limits")
async def effective_limits() -> dict[str, object]:
    """The limits in force, as the running process reads them.

    The numbers live in environment variables, which means the only way to know what a
    deployment is actually enforcing was to read its env file. This answers that from the
    outside: what is on, and what each limit is, without exposing a secret.
    """
    from app.ai.execution import limits

    return {
        "chat": {
            "enabled": limits.limits_enabled(),
            "per_minute": limits.chat_per_minute(),
            "burst": limits.chat_burst(),
            "daily_quota": limits.daily_quota(),
        },
        "api": {
            "enabled": limits.api_limits_enabled(),
            "per_minute": limits.api_per_minute(),
            "burst": limits.api_burst(),
        },
        "configured_by": [
            "INDOONE_CHAT_LIMITS_ENABLED",
            "INDOONE_CHAT_RATE_PER_MINUTE",
            "INDOONE_CHAT_RATE_BURST",
            "INDOONE_CHAT_DAILY_QUOTA",
            "INDOONE_API_LIMITS_ENABLED",
            "INDOONE_API_RATE_PER_MINUTE",
            "INDOONE_API_RATE_BURST",
        ],
        "documented_in": ".env.example",
    }


@router.get("/observability")
async def observability_snapshot(
    x_indoone_observability_token: str = Header(default=""),
) -> dict[str, object]:
    """Return the aggregate spans, tool stats, model stats and cost."""

    _require_token(x_indoone_observability_token)
    return snapshot()


@router.post("/observability/reset")
async def observability_reset(
    x_indoone_observability_token: str = Header(default=""),
) -> dict[str, str]:
    """Clear the aggregates, so the next window is measured on its own."""

    _require_token(x_indoone_observability_token)
    reset()
    return {"status": "reset"}
