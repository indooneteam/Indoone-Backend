from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx

from app.capabilities.store import consume_oauth_state, get_integration_token, upsert_integration_token
from app.capabilities.youtube import _auth_headers, _fernet

_API_BASE = "https://youtubeanalytics.googleapis.com/v2"
_OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_INTEGRATION_ID = "youtube_analytics"
_YOUTUBE_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
_ANALYTICS_READ_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
_MAX_RESULTS = 200
_MAX_START_INDEX = 100_000
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_,-]+$")
_FILTER_PATTERN = re.compile(r"^[A-Za-z0-9_.,;=:+\-]+$")
_CURRENCY_PATTERN = re.compile(r"^[A-Za-z]{3}$")
_DATE_FORMAT = "%Y-%m-%d"


def build_youtube_analytics_authorization(state: str, redirect_uri: str) -> dict[str, object]:
    client_id = _client_id()
    scope = f"{_YOUTUBE_READ_SCOPE} {_ANALYTICS_READ_SCOPE}"
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri.strip(),
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "scope": scope,
        "state": state.strip(),
    }
    return {
        "integration": "youtube",
        "connect_mode": "analytics",
        "authorization_url": f"{_OAUTH_URL}?{urlencode(params)}",
        "scope": scope,
        "capabilities": [
            "view channel analytics",
            "view daily trends",
            "view top video performance",
            "view traffic source and country breakdowns",
            "run custom read-only analytics reports",
        ],
        "secrets_exposed": False,
    }


def _client_id() -> str:
    import os

    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    if not client_id or not os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip():
        raise RuntimeError("google oauth client credentials are not configured")
    return client_id


def _client_credentials() -> tuple[str, str]:
    import os

    client_id = os.getenv("INDOONE_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("INDOONE_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("google oauth client credentials are not configured")
    return client_id, client_secret


def _store_token(user_id: str, body: dict[str, object], fallback_refresh: str | None = None) -> None:
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("youtube analytics oauth token response contains no access token")
    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = fallback_refresh
    expires_at: str | None = None
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
    scope = str(body.get("scope") or f"{_YOUTUBE_READ_SCOPE} {_ANALYTICS_READ_SCOPE}")
    upsert_integration_token(
        user_id,
        _INTEGRATION_ID,
        _fernet().encrypt(access_token.encode("utf-8")),
        _fernet().encrypt(refresh_token.encode("utf-8")) if refresh_token else None,
        str(body.get("token_type") or "Bearer"),
        scope,
        expires_at,
    )


async def exchange_youtube_analytics_code(state: str, code: str) -> dict[str, object]:
    state_data = consume_oauth_state(state.strip(), _INTEGRATION_ID)
    if state_data is None:
        raise ValueError("invalid or expired youtube analytics oauth state")
    client_id, client_secret = _client_credentials()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code.strip(),
                "redirect_uri": state_data["redirect_uri"],
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube analytics oauth returned an invalid response")
    _validate_granted_scopes(str(body.get("scope") or ""))
    _store_token(state_data["user_id"], body)
    return {
        "integration": "youtube",
        "user_id": state_data["user_id"],
        "connected": True,
        "connect_mode": "analytics",
        "scope": str(body.get("scope") or ""),
        "secrets_exposed": False,
    }


def _validate_granted_scopes(scope: str) -> str:
    granted = set(scope.split())
    required = {_YOUTUBE_READ_SCOPE, _ANALYTICS_READ_SCOPE}
    if not required.issubset(granted):
        raise PermissionError("youtube analytics access requires youtube.readonly and yt-analytics.readonly scopes")
    return scope


def _token_row(user_id: str) -> dict[str, object]:
    row = get_integration_token(user_id.strip(), _INTEGRATION_ID)
    if row is None:
        raise ValueError("youtube analytics is not connected for user")
    _validate_granted_scopes(str(row.get("scope") or ""))
    return row


async def _refresh(user_id: str, row: dict[str, object]) -> str:
    refresh_raw = row.get("refresh_token")
    if refresh_raw is None:
        raise RuntimeError("youtube analytics access token expired and no refresh token is stored")
    try:
        refresh_token = _fernet().decrypt(bytes(refresh_raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored youtube analytics refresh token cannot be decrypted") from exc
    client_id, client_secret = _client_credentials()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube analytics oauth refresh returned an invalid response")
    scope = str(body.get("scope") or row.get("scope") or "")
    _validate_granted_scopes(scope)
    _store_token(user_id, body, fallback_refresh=refresh_token)
    refreshed = _token_row(user_id)
    raw = refreshed.get("access_token")
    if raw is None:
        raise RuntimeError("refreshed youtube analytics access token is empty")
    try:
        return _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("refreshed youtube analytics access token cannot be decrypted") from exc


async def _access_token(user_id: str) -> str:
    row = _token_row(user_id)
    raw = row.get("access_token")
    if raw is None:
        raise RuntimeError("stored youtube analytics access token is empty")
    try:
        token = _fernet().decrypt(bytes(raw)).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored youtube analytics access token cannot be decrypted") from exc
    expires_at = row.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc) + timedelta(seconds=30):
                return await _refresh(user_id, row)
        except ValueError:
            pass
    return token


def _validate_date(value: str, field_name: str) -> str:
    normalized = value.strip()
    try:
        datetime.strptime(normalized, _DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc
    return normalized


def _validate_window(start_date: str, end_date: str) -> tuple[str, str]:
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return start, end


def _validate_tokens(value: str, field_name: str) -> str:
    normalized = ",".join(part.strip() for part in value.split(",") if part.strip())
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > 2000 or not _TOKEN_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _validate_filters(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 5000 or not _FILTER_PATTERN.fullmatch(normalized):
        raise ValueError("filters contains unsupported characters")
    return normalized


def _validate_currency(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    if not _CURRENCY_PATTERN.fullmatch(normalized):
        raise ValueError("currency must be a 3-letter ISO code")
    return normalized


def _build_report_params(
    *,
    start_date: str,
    end_date: str,
    metrics: str,
    dimensions: str = "",
    filters: str | None = None,
    sort: str = "",
    max_results: int = 50,
    start_index: int = 1,
    currency: str | None = None,
    ids: str = "channel==MINE",
) -> dict[str, object]:
    start, end = _validate_window(start_date, end_date)
    if ids.strip() != "channel==MINE":
        raise ValueError("ids must be channel==MINE")
    if not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError(f"max_results must be between 1 and {_MAX_RESULTS}")
    if not 1 <= start_index <= _MAX_START_INDEX:
        raise ValueError(f"start_index must be between 1 and {_MAX_START_INDEX}")
    metrics = _validate_tokens(metrics, "metrics")
    dimensions = _validate_tokens(dimensions, "dimensions") if dimensions.strip() else ""
    sort = _validate_tokens(sort, "sort") if sort.strip() else ""
    normalized_filters = _validate_filters(filters)
    normalized_currency = _validate_currency(currency)
    params: dict[str, object] = {
        "ids": ids.strip(),
        "startDate": start,
        "endDate": end,
        "metrics": metrics,
        "maxResults": max_results,
        "startIndex": start_index,
    }
    if dimensions:
        params["dimensions"] = dimensions
    if normalized_filters:
        params["filters"] = normalized_filters
    if sort:
        params["sort"] = sort
    if normalized_currency:
        params["currency"] = normalized_currency
    return params


async def _request_report(user_id: str, params: dict[str, object]) -> dict[str, object]:
    token = await _access_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{_API_BASE}/reports",
            headers=_auth_headers(token),
            params=params,
        )
        if response.status_code == 401:
            row = _token_row(user_id)
            token = await _refresh(user_id, row)
            response = await client.get(
                f"{_API_BASE}/reports",
                headers=_auth_headers(token),
                params=params,
            )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube analytics returned an invalid report response")
    return body


def _normalize_report(body: dict[str, object], params: dict[str, object]) -> dict[str, object]:
    headers = body.get("columnHeaders") if isinstance(body.get("columnHeaders"), list) else []
    rows = body.get("rows") if isinstance(body.get("rows"), list) else []
    header_names = [
        str(item.get("name"))
        for item in headers
        if isinstance(item, dict) and item.get("name")
    ]
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        normalized_rows.append({
            name: row[index] if index < len(row) else None
            for index, name in enumerate(header_names)
        })
    return {
        "integration": "youtube",
        "operation": "analytics_report",
        "headers": headers,
        "rows": rows,
        "rows_as_objects": normalized_rows,
        "row_count": len(normalized_rows),
        "query": params,
        "secrets_exposed": False,
    }


async def query_report(
    user_id: str,
    *,
    start_date: str,
    end_date: str,
    metrics: str,
    dimensions: str = "",
    filters: str | None = None,
    sort: str = "",
    max_results: int = 50,
    start_index: int = 1,
    currency: str | None = None,
) -> dict[str, object]:
    params = _build_report_params(
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        sort=sort,
        max_results=max_results,
        start_index=start_index,
        currency=currency,
    )
    result = await _request_report(user_id, params)
    return _normalize_report(result, params)


async def channel_overview(user_id: str, start_date: str, end_date: str) -> dict[str, object]:
    return await query_report(
        user_id,
        start_date=start_date,
        end_date=end_date,
        metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments,shares,subscribersGained,subscribersLost",
    )


async def daily_channel_trend(user_id: str, start_date: str, end_date: str, max_results: int = 200) -> dict[str, object]:
    return await query_report(
        user_id,
        start_date=start_date,
        end_date=end_date,
        metrics="views,estimatedMinutesWatched,likes,comments,subscribersGained,subscribersLost",
        dimensions="day",
        sort="day",
        max_results=max_results,
    )


async def top_videos(user_id: str, start_date: str, end_date: str, max_results: int = 20) -> dict[str, object]:
    return await query_report(
        user_id,
        start_date=start_date,
        end_date=end_date,
        metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments,shares,subscribersGained,subscribersLost",
        dimensions="video",
        sort="-views",
        max_results=max_results,
    )


async def traffic_sources(user_id: str, start_date: str, end_date: str, max_results: int = 50) -> dict[str, object]:
    return await query_report(
        user_id,
        start_date=start_date,
        end_date=end_date,
        metrics="views,estimatedMinutesWatched",
        dimensions="insightTrafficSourceType",
        sort="-views",
        max_results=max_results,
    )


async def country_breakdown(user_id: str, start_date: str, end_date: str, max_results: int = 50) -> dict[str, object]:
    return await query_report(
        user_id,
        start_date=start_date,
        end_date=end_date,
        metrics="views,estimatedMinutesWatched,likes,comments",
        dimensions="country",
        sort="-views",
        max_results=max_results,
    )
