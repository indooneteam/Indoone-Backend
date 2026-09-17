from __future__ import annotations

import base64
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.google_photos import (
    build_google_photos_authorization,
    create_album,
    exchange_google_photos_code,
    get_media_item,
    list_albums,
    list_media,
    search_media,
    upload_media,
)
from app.capabilities.store import create_oauth_state

router = APIRouter(prefix="/google-photos", tags=["google-photos"])


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)
    include_upload: bool = True


class CallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class ListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    page_size: int = Field(default=25, ge=1, le=100)
    page_token: str = Field(default="", max_length=4096)


class SearchRequest(ListRequest):
    album_id: str = Field(default="", max_length=512)
    media_type: str = Field(default="", max_length=32)
    order: str = Field(default="", max_length=64)


class ItemRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    media_item_id: str = Field(min_length=1, max_length=512)


class AlbumCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=500)
    approved: bool = False


class UploadRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    filename: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=1, max_length=256)
    content_base64: str = Field(min_length=1, max_length=40_000_000)
    description: str = Field(default="", max_length=1000)
    album_id: str = Field(default="", max_length=512)
    approved: bool = False


@router.post("/connect")
async def connect(request: ConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "google_photos", request.redirect_uri.strip())
        return {**build_google_photos_authorization(state, request.redirect_uri, request.include_upload), "state": state}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: CallbackRequest) -> dict[str, object]:
    try:
        return await exchange_google_photos_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"google photos oauth failed: {exc}") from exc


@router.post("/media")
async def media(request: ListRequest) -> dict[str, object]:
    try:
        return await list_media(request.user_id, request.page_size, request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/search")
async def search(request: SearchRequest) -> dict[str, object]:
    try:
        return await search_media(request.user_id, request.page_size, request.page_token, request.album_id, request.media_type, request.order)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/media-item")
async def media_item(request: ItemRequest) -> dict[str, object]:
    try:
        return await get_media_item(request.user_id, request.media_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/albums")
async def albums(request: ListRequest) -> dict[str, object]:
    try:
        return await list_albums(request.user_id, min(request.page_size, 50), request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/create-album")
async def create_album_endpoint(request: AlbumCreateRequest) -> dict[str, object]:
    try:
        return await create_album(request.user_id, request.title, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/upload")
async def upload(request: UploadRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        return await upload_media(
            request.user_id,
            content,
            request.filename,
            request.mime_type,
            request.description,
            request.album_id,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google photos provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
