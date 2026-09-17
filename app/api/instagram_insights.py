from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_insights import get_account_insights, get_media_insights

router = APIRouter(prefix="/integrations/instagram/insights", tags=["instagram-insights"])


class AccountInsightsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    metrics: list[str] = Field(min_length=1, max_length=50)
    period: str = Field(default="day", min_length=1, max_length=64)
    since: int | None = Field(default=None, ge=0)
    until: int | None = Field(default=None, ge=0)


class MediaInsightsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    media_id: str = Field(min_length=1, max_length=256)
    metrics: list[str] = Field(min_length=1, max_length=50)


@router.post("/account")
async def account_insights(request: AccountInsightsRequest) -> dict[str, object]:
    try:
        return await get_account_insights(
            request.user_id,
            request.metrics,
            request.period,
            request.since,
            request.until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram insights provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/media")
async def media_insights(request: MediaInsightsRequest) -> dict[str, object]:
    try:
        return await get_media_insights(request.user_id, request.media_id, request.metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram insights provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
