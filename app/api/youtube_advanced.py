from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.youtube_advanced import (
    delete_caption,
    download_caption,
    get_video_advanced,
    insert_caption,
    list_captions,
    list_my_shorts,
    update_caption,
    update_video_advanced,
)

router = APIRouter(prefix="/youtube/advanced", tags=["youtube-advanced"])


class UserVideoRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)


class ShortListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=20, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class AdvancedVideoUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=5000)
    category_id: str | None = Field(default=None, max_length=16)
    tags: list[str] | None = Field(default=None, max_length=500)
    default_language: str | None = Field(default=None, max_length=35)
    privacy_status: str | None = Field(default=None, max_length=32)
    embeddable: bool | None = None
    license: str | None = Field(default=None, max_length=32)
    public_stats_viewable: bool | None = None
    publish_at: str | None = Field(default=None, max_length=64)
    self_declared_made_for_kids: bool | None = None
    contains_synthetic_media: bool | None = None
    recording_date: str | None = Field(default=None, max_length=64)
    localizations: dict[str, dict[str, str]] | None = None
    approved: bool = False


class CaptionInsertRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)
    language: str = Field(min_length=1, max_length=35)
    name: str = Field(min_length=1, max_length=150)
    content_base64: str = Field(min_length=1, max_length=70_000_000)
    mime_type: str = Field(default="application/octet-stream", max_length=128)
    track_kind: str = Field(default="standard", max_length=16)
    is_draft: bool = False
    approved: bool = False


class CaptionUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    caption_id: str = Field(min_length=1, max_length=128)
    content_base64: str | None = Field(default=None, max_length=70_000_000)
    mime_type: str = Field(default="application/octet-stream", max_length=128)
    is_draft: bool | None = None
    approved: bool = False


class CaptionDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    caption_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class CaptionDownloadRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    caption_id: str = Field(min_length=1, max_length=128)
    tfmt: str | None = Field(default=None, max_length=16)
    tlang: str | None = Field(default=None, max_length=35)


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "integration": "youtube",
        "mode": "advanced_video",
        "oauth_connect_mode": "channel_manager",
        "shorts": [
            "inspect Shorts eligibility from processed video dimensions and duration",
            "list authenticated-channel videos that currently meet Shorts eligibility rules",
        ],
        "video_operations": [
            "inspect processing and file details",
            "update scheduled publish time",
            "update localized titles and descriptions",
            "update synthetic-media and made-for-kids declarations",
            "update recording date and advanced video status metadata",
        ],
        "captions": [
            "list caption tracks",
            "insert caption tracks",
            "update caption tracks",
            "download caption tracks",
            "delete caption tracks",
        ],
        "write_operations_require_approval": True,
        "secrets_exposed": False,
    }


def _route_error(exc: Exception, operation: str) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (httpx.HTTPError, RuntimeError)):
        return HTTPException(status_code=502, detail=f"youtube advanced {operation} failed: {exc}")
    return HTTPException(status_code=500, detail=f"youtube advanced {operation} failed")


@router.post("/shorts")
async def shorts(request: ShortListRequest) -> dict[str, object]:
    try:
        return await list_my_shorts(request.user_id, request.max_results, request.page_token)
    except Exception as exc:
        raise _route_error(exc, "shorts list") from exc


@router.post("/video")
async def video(request: UserVideoRequest) -> dict[str, object]:
    try:
        return await get_video_advanced(request.user_id, request.video_id)
    except Exception as exc:
        raise _route_error(exc, "video inspection") from exc


@router.post("/video/update")
async def video_update(request: AdvancedVideoUpdateRequest) -> dict[str, object]:
    try:
        return await update_video_advanced(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "advanced video update") from exc


@router.post("/captions")
async def captions(request: UserVideoRequest) -> dict[str, object]:
    try:
        return await list_captions(request.user_id, request.video_id)
    except Exception as exc:
        raise _route_error(exc, "caption list") from exc


@router.post("/caption/insert")
async def caption_insert(request: CaptionInsertRequest) -> dict[str, object]:
    try:
        return await insert_caption(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "caption insert") from exc


@router.post("/caption/update")
async def caption_update(request: CaptionUpdateRequest) -> dict[str, object]:
    try:
        return await update_caption(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "caption update") from exc


@router.post("/caption/delete")
async def caption_delete(request: CaptionDeleteRequest) -> dict[str, object]:
    try:
        return await delete_caption(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "caption delete") from exc


@router.post("/caption/download")
async def caption_download(request: CaptionDownloadRequest) -> dict[str, object]:
    try:
        return await download_caption(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "caption download") from exc
