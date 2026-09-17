from __future__ import annotations

import re
from typing import Any

import httpx

from app.capabilities.instagram import (
    _GRAPH_URL,
    _access_token,
    _refresh_access_token,
    _token_row,
)

_METRIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_PERIOD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _validate_metrics(metrics: list[str]) -> list[str]:
    if not metrics:
        raise ValueError("at least one insight metric is required")
    if len(metrics) > 50:
        raise ValueError("at most 50 insight metrics may be requested")
    normalized = []
    seen: set[str] = set()
    for metric in metrics:
        value = metric.strip()
        if not _METRIC_RE.fullmatch(value):
            raise ValueError("insight metrics must contain only letters, numbers, and underscores")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _validate_period(period: str) -> str:
    value = period.strip()
    if not value or not _PERIOD_RE.fullmatch(value):
        raise ValueError("period must be a valid Instagram insights period")
    return value


def _validate_range(since: int | None, until: int | None) -> None:
    if since is not None and since < 0:
        raise ValueError("since must be a non-negative Unix timestamp")
    if until is not None and until < 0:
        raise ValueError("until must be a non-negative Unix timestamp")
    if since is not None and until is not None and since > until:
        raise ValueError("since must be less than or equal to until")


async def _get_json_with_retry(
    user_id: str,
    path: str,
    params: dict[str, object],
) -> dict[str, Any]:
    token = await _access_token(user_id)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GRAPH_URL}/{path.lstrip('/')}", params=params, headers=headers)
        if response.status_code == 401:
            refreshed = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.get(
                f"{_GRAPH_URL}/{path.lstrip('/')}",
                params=params,
                headers={"Authorization": f"Bearer {refreshed}", "Accept": "application/json"},
            )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram insights response is invalid")
    return body


async def get_account_insights(
    user_id: str,
    metrics: list[str],
    period: str = "day",
    since: int | None = None,
    until: int | None = None,
) -> dict[str, object]:
    normalized_metrics = _validate_metrics(metrics)
    normalized_period = _validate_period(period)
    _validate_range(since, until)
    params: dict[str, object] = {
        "metric": ",".join(normalized_metrics),
        "period": normalized_period,
    }
    if since is not None:
        params["since"] = since
    if until is not None:
        params["until"] = until
    body = await _get_json_with_retry(user_id, "me/insights", params)
    return {
        "integration": "instagram",
        "scope": "account",
        "metrics": normalized_metrics,
        "period": normalized_period,
        "since": since,
        "until": until,
        "data": body.get("data", []),
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }


async def get_media_insights(user_id: str, media_id: str, metrics: list[str]) -> dict[str, object]:
    media = media_id.strip()
    if not media:
        raise ValueError("media_id is required")
    normalized_metrics = _validate_metrics(metrics)
    body = await _get_json_with_retry(user_id, f"{media}/insights", {"metric": ",".join(normalized_metrics)})
    return {
        "integration": "instagram",
        "scope": "media",
        "media_id": media,
        "metrics": normalized_metrics,
        "data": body.get("data", []),
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }
