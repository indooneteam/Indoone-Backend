from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.instagram import (
    build_instagram_authorization,
    create_media_container,
    exchange_instagram_code,
    get_profile,
    list_media,
    publish_media,
)
from app.capabilities.store import create_oauth_state

router = APIRouter(prefix="/integrations/instagram", tags=["instagram"])


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)
    include_publishing: bool = True
    include_messages: bool = False
    include_comments: bool = False


class CallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class UserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class MediaListRequest(UserRequest):
    limit: int = Field(default=25, ge=1, le=100)
    after: str = Field(default="", max_length=2048)


class CreateContainerRequest(UserRequest):
    image_url: str = Field(default="", max_length=4000)
    video_url: str = Field(default="", max_length=4000)
    caption: str = Field(default="", max_length=2200)
    media_type: str = Field(default="IMAGE", max_length=32)


class PublishRequest(UserRequest):
    creation_id: str = Field(min_length=1, max_length=512)
    approved: bool = False


@router.post("/connect")
async def connect(request: ConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "instagram", request.redirect_uri.strip())
        return {
            **build_instagram_authorization(
                state,
                request.redirect_uri,
                request.include_publishing,
                request.include_messages,
                request.include_comments,
            ),
            "state": state,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: CallbackRequest) -> dict[str, object]:
    try:
        return await exchange_instagram_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"instagram oauth failed: {exc}") from exc


@router.post("/profile")
async def profile(request: UserRequest) -> dict[str, object]:
    try:
        return await get_profile(request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/media")
async def media(request: MediaListRequest) -> dict[str, object]:
    try:
        return await list_media(request.user_id, request.limit, request.after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/create-container")
async def create_container(request: CreateContainerRequest) -> dict[str, object]:
    try:
        return await create_media_container(
            request.user_id,
            request.image_url,
            request.video_url,
            request.caption,
            request.media_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/publish")
async def publish(request: PublishRequest) -> dict[str, object]:
    try:
        return await publish_media(request.user_id, request.creation_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
