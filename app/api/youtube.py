from __future__ import annotations

import base64
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.store import create_oauth_state
from app.capabilities.youtube import (
    build_youtube_authorization,
    exchange_youtube_code,
    get_channel_dashboard,
    get_my_channel,
    get_videos,
    search_youtube,
    upload_video,
    youtube_connect_capabilities,
)

router = APIRouter(prefix="/youtube", tags=["youtube"])


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)
    connect_mode: str = Field(default="channel", max_length=32)
    read_only: bool | None = None


class CallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    resource_type: str = Field(default="video", max_length=32)
    max_results: int = Field(default=10, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class VideoRequest(BaseModel):
    video_ids: list[str] = Field(min_length=1, max_length=50)
    user_id: str = Field(default="", max_length=256)


class ChannelRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class DashboardRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=20, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class UploadRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=5000)
    privacy_status: str = Field(default="private", max_length=32)
    category_id: str = Field(default="22", max_length=16)
    mime_type: str = Field(default="video/mp4", max_length=128)
    content_base64: str = Field(min_length=1, max_length=350_000_000)
    approved: bool = False


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "integration": "youtube",
        "modes": {
            "normal": youtube_connect_capabilities("normal"),
            "channel": youtube_connect_capabilities("channel"),
        },
        "secrets_exposed": False,
    }


def _create_connect_response(request: ConnectRequest, mode: str) -> dict[str, object]:
    state = token_urlsafe(32)
    create_oauth_state(state, request.user_id, "youtube", request.redirect_uri.strip())
    return {
        **build_youtube_authorization(
            state,
            request.redirect_uri,
            read_only=(mode == "normal"),
            connect_mode=mode,
        ),
        "state": state,
    }


@router.post("/connect")
async def connect(request: ConnectRequest) -> dict[str, object]:
    requested_mode = request.connect_mode.strip().lower()
    if request.read_only is not None:
        requested_mode = "normal" if request.read_only else "channel"
    try:
        return _create_connect_response(request, requested_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/normal/connect")
async def normal_connect(request: ConnectRequest) -> dict[str, object]:
    try:
        return _create_connect_response(request, "normal")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/channel/connect")
async def channel_connect(request: ConnectRequest) -> dict[str, object]:
    try:
        return _create_connect_response(request, "channel")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: CallbackRequest) -> dict[str, object]:
    try:
        return await exchange_youtube_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube oauth failed: {exc}") from exc


@router.post("/search")
async def search(request: SearchRequest) -> dict[str, object]:
    try:
        return await search_youtube(request.query, request.resource_type, request.max_results, request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube search failed: {exc}") from exc


@router.post("/videos")
async def videos(request: VideoRequest) -> dict[str, object]:
    try:
        return await get_videos(request.video_ids, request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube video lookup failed: {exc}") from exc


@router.post("/my-channel")
async def my_channel(request: ChannelRequest) -> dict[str, object]:
    try:
        return await get_my_channel(request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube channel lookup failed: {exc}") from exc


@router.post("/dashboard")
async def dashboard(request: DashboardRequest) -> dict[str, object]:
    try:
        return await get_channel_dashboard(request.user_id, request.max_results, request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube dashboard failed: {exc}") from exc


@router.post("/upload")
async def upload(request: UploadRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        return await upload_video(
            request.user_id,
            content,
            request.title,
            request.description,
            request.privacy_status,
            request.category_id,
            request.mime_type,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube upload failed: {exc}") from exc
