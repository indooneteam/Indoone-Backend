from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.store import create_oauth_state
from app.capabilities.youtube_analytics import (
    build_youtube_analytics_authorization,
    channel_overview,
    country_breakdown,
    daily_channel_trend,
    exchange_youtube_analytics_code,
    query_report,
    top_videos,
    traffic_sources,
)

router = APIRouter(prefix="/youtube/analytics", tags=["youtube-analytics"])


class AnalyticsConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)


class AnalyticsCallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class AnalyticsReportRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    start_date: str = Field(min_length=10, max_length=10)
    end_date: str = Field(min_length=10, max_length=10)
    metrics: str = Field(min_length=1, max_length=2000)
    dimensions: str = Field(default="", max_length=2000)
    filters: str | None = Field(default=None, max_length=5000)
    sort: str = Field(default="", max_length=1000)
    max_results: int = Field(default=50, ge=1, le=200)
    start_index: int = Field(default=1, ge=1, le=100000)
    currency: str | None = Field(default=None, max_length=3)


class AnalyticsWindowRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    start_date: str = Field(min_length=10, max_length=10)
    end_date: str = Field(min_length=10, max_length=10)
    max_results: int = Field(default=50, ge=1, le=200)


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "integration": "youtube",
        "mode": "analytics",
        "capabilities": [
            "view channel analytics",
            "view daily trends",
            "view top video performance",
            "view traffic source and country breakdowns",
            "run custom read-only analytics reports",
        ],
        "write_operations": [],
        "secrets_exposed": False,
    }


@router.post("/connect")
async def connect(request: AnalyticsConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "youtube_analytics", request.redirect_uri.strip())
        return {**build_youtube_analytics_authorization(state, request.redirect_uri), "state": state}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: AnalyticsCallbackRequest) -> dict[str, object]:
    try:
        return await exchange_youtube_analytics_code(request.state, request.code)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube analytics oauth failed: {exc}") from exc


def _report_error(exc: Exception, operation: str) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, (httpx.HTTPError, RuntimeError)):
        return HTTPException(status_code=502, detail=f"youtube analytics {operation} failed: {exc}")
    return HTTPException(status_code=500, detail=f"youtube analytics {operation} failed")


@router.post("/report")
async def report(request: AnalyticsReportRequest) -> dict[str, object]:
    try:
        return await query_report(
            request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            metrics=request.metrics,
            dimensions=request.dimensions,
            filters=request.filters,
            sort=request.sort,
            max_results=request.max_results,
            start_index=request.start_index,
            currency=request.currency,
        )
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "report") from exc


@router.post("/overview")
async def overview(request: AnalyticsWindowRequest) -> dict[str, object]:
    try:
        return await channel_overview(request.user_id, request.start_date, request.end_date)
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "overview") from exc


@router.post("/daily")
async def daily(request: AnalyticsWindowRequest) -> dict[str, object]:
    try:
        return await daily_channel_trend(request.user_id, request.start_date, request.end_date, request.max_results)
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "daily trend") from exc


@router.post("/top-videos")
async def top_videos_route(request: AnalyticsWindowRequest) -> dict[str, object]:
    try:
        return await top_videos(request.user_id, request.start_date, request.end_date, request.max_results)
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "top videos") from exc


@router.post("/traffic-sources")
async def traffic_sources_route(request: AnalyticsWindowRequest) -> dict[str, object]:
    try:
        return await traffic_sources(request.user_id, request.start_date, request.end_date, request.max_results)
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "traffic sources") from exc


@router.post("/countries")
async def countries(request: AnalyticsWindowRequest) -> dict[str, object]:
    try:
        return await country_breakdown(request.user_id, request.start_date, request.end_date, request.max_results)
    except (ValueError, PermissionError, httpx.HTTPError, RuntimeError) as exc:
        raise _report_error(exc, "country breakdown") from exc
