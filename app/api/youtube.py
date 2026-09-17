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
from app.capabilities.youtube_video_manager import (
    build_youtube_manager_authorization,
    delete_video,
    exchange_youtube_manager_code,
    set_video_thumbnail,
    update_video,
)
from app.capabilities.youtube_playlist_manager import (
    add_video_to_playlist,
    create_playlist,
    delete_playlist,
    list_playlist_items,
    list_playlists,
    remove_playlist_item,
    reorder_playlist_item,
    update_playlist,
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


class ManagerConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)


class ManagerCallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class VideoUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    category_id: str | None = Field(default=None, max_length=16)
    privacy_status: str | None = Field(default=None, max_length=32)
    tags: list[str] | None = Field(default=None, max_length=500)
    approved: bool = False


class VideoDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class ThumbnailRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    video_id: str = Field(min_length=1, max_length=128)
    mime_type: str = Field(min_length=1, max_length=64)
    content_base64: str = Field(min_length=1, max_length=70_000_000)
    approved: bool = False


class PlaylistListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=20, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class PlaylistRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_id: str = Field(min_length=1, max_length=128)


class PlaylistCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=150)
    description: str = Field(default="", max_length=5000)
    privacy_status: str = Field(default="private", max_length=32)
    approved: bool = False


class PlaylistUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=150)
    description: str | None = Field(default=None, max_length=5000)
    privacy_status: str | None = Field(default=None, max_length=32)
    approved: bool = False


class PlaylistItemListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_id: str = Field(min_length=1, max_length=128)
    max_results: int = Field(default=50, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class PlaylistAddRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_id: str = Field(min_length=1, max_length=128)
    video_id: str = Field(min_length=1, max_length=128)
    position: int | None = Field(default=None, ge=0, le=10000)
    approved: bool = False


class PlaylistItemRemoveRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_item_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class PlaylistItemReorderRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    playlist_item_id: str = Field(min_length=1, max_length=128)
    position: int = Field(ge=0, le=10000)
    approved: bool = False


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "integration": "youtube",
        "modes": {
            "normal": youtube_connect_capabilities("normal"),
            "channel": youtube_connect_capabilities("channel"),
            "channel_manager": [
                "edit video title, description, category, tags, and privacy",
                "delete videos with explicit approval",
                "set custom video thumbnails with explicit approval",
                "list, create, edit, and delete playlists with explicit approval for writes",
                "list playlist videos and add, remove, or reorder them with explicit approval for writes",
            ],
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


@router.post("/video-manager/connect")
async def video_manager_connect(request: ManagerConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "youtube_video_manager", request.redirect_uri.strip())
        return {**build_youtube_manager_authorization(state, request.redirect_uri), "state": state}
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


@router.post("/video-manager/callback")
async def video_manager_callback(request: ManagerCallbackRequest) -> dict[str, object]:
    try:
        return await exchange_youtube_manager_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube video manager oauth failed: {exc}") from exc


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


@router.post("/video/update")
async def video_update(request: VideoUpdateRequest) -> dict[str, object]:
    try:
        return await update_video(
            request.user_id,
            request.video_id,
            title=request.title,
            description=request.description,
            category_id=request.category_id,
            privacy_status=request.privacy_status,
            tags=request.tags,
            approved=request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube video update failed: {exc}") from exc


@router.post("/video/delete")
async def video_delete(request: VideoDeleteRequest) -> dict[str, object]:
    try:
        return await delete_video(request.user_id, request.video_id, approved=request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube video delete failed: {exc}") from exc


@router.post("/video/thumbnail")
async def video_thumbnail(request: ThumbnailRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        return await set_video_thumbnail(
            request.user_id,
            request.video_id,
            content,
            request.mime_type,
            approved=request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube thumbnail update failed: {exc}") from exc


@router.post("/playlist/list")
async def playlist_list(request: PlaylistListRequest) -> dict[str, object]:
    try:
        return await list_playlists(request.user_id, request.max_results, request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist list failed: {exc}") from exc


@router.post("/playlist/get")
async def playlist_get(request: PlaylistRequest) -> dict[str, object]:
    try:
        return await __import__("app.capabilities.youtube_playlist_manager", fromlist=["get_playlist"]).get_playlist(request.user_id, request.playlist_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist get failed: {exc}") from exc


@router.post("/playlist/create")
async def playlist_create(request: PlaylistCreateRequest) -> dict[str, object]:
    try:
        return await create_playlist(
            request.user_id,
            request.title,
            request.description,
            request.privacy_status,
            approved=request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist create failed: {exc}") from exc


@router.post("/playlist/update")
async def playlist_update(request: PlaylistUpdateRequest) -> dict[str, object]:
    try:
        return await update_playlist(
            request.user_id,
            request.playlist_id,
            title=request.title,
            description=request.description,
            privacy_status=request.privacy_status,
            approved=request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist update failed: {exc}") from exc


@router.post("/playlist/delete")
async def playlist_delete(request: PlaylistRequest) -> dict[str, object]:
    try:
        return await delete_playlist(request.user_id, request.playlist_id, approved=True)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist delete failed: {exc}") from exc


@router.post("/playlist/items")
async def playlist_items(request: PlaylistItemListRequest) -> dict[str, object]:
    try:
        return await list_playlist_items(request.user_id, request.playlist_id, request.max_results, request.page_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist items failed: {exc}") from exc


@router.post("/playlist/add")
async def playlist_add(request: PlaylistAddRequest) -> dict[str, object]:
    try:
        return await add_video_to_playlist(request.user_id, request.playlist_id, request.video_id, request.position, approved=request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist add failed: {exc}") from exc


@router.post("/playlist/remove")
async def playlist_remove(request: PlaylistItemRemoveRequest) -> dict[str, object]:
    try:
        return await remove_playlist_item(request.user_id, request.playlist_item_id, approved=request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist remove failed: {exc}") from exc


@router.post("/playlist/reorder")
async def playlist_reorder(request: PlaylistItemReorderRequest) -> dict[str, object]:
    try:
        return await reorder_playlist_item(request.user_id, request.playlist_item_id, request.position, approved=request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube playlist reorder failed: {exc}") from exc


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
