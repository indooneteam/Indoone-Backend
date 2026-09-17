from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_advanced import (
    get_container_status,
    get_media_details,
    list_reels,
    list_stories,
)

router = APIRouter(prefix="/integrations/instagram/advanced", tags=["instagram-advanced"])


class MediaListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=25, ge=1, le=100)
    after: str = Field(default="", max_length=2048)


class MediaDetailsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    media_id: str = Field(min_length=1, max_length=256)


class ContainerRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    creation_id: str = Field(min_length=1, max_length=256)


@router.post("/reels")
async def reels(request: MediaListRequest) -> dict[str, object]:
    try:
        return await list_reels(request.user_id, request.limit, request.after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram reels provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/stories")
async def stories(request: MediaListRequest) -> dict[str, object]:
    try:
        return await list_stories(request.user_id, request.limit, request.after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram stories provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/media")
async def media(request: MediaDetailsRequest) -> dict[str, object]:
    try:
        return await get_media_details(request.user_id, request.media_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram media provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/container")
async def container(request: ContainerRequest) -> dict[str, object]:
    try:
        return await get_container_status(request.user_id, request.creation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram container provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
