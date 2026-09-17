from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.store import create_oauth_state
from app.capabilities.youtube_live import (
    bind_broadcast,
    create_broadcast,
    create_stream,
    delete_broadcast,
    delete_live_chat_message,
    delete_stream,
    get_broadcast,
    get_stream,
    list_broadcasts,
    list_live_chat_messages,
    list_streams,
    send_live_chat_message,
    transition_broadcast,
    update_broadcast,
    update_stream,
)
from app.capabilities.youtube_video_manager import (
    build_youtube_manager_authorization,
    exchange_youtube_manager_code,
)

router = APIRouter(prefix="/youtube/live", tags=["youtube-live"])


class LiveConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)


class LiveCallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class BroadcastListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_status: str | None = Field(default=None, max_length=32)
    broadcast_id: str | None = Field(default=None, max_length=128)
    max_results: int = Field(default=20, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class BroadcastGetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_id: str = Field(min_length=1, max_length=128)


class BroadcastCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=100)
    scheduled_start_time: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=5000)
    scheduled_end_time: str | None = Field(default=None, max_length=64)
    privacy_status: str = Field(default="private", max_length=32)
    category_id: str | None = Field(default=None, max_length=16)
    enable_dvr: bool | None = None
    record_from_start: bool | None = None
    enable_auto_start: bool | None = None
    enable_auto_stop: bool | None = None
    latency_preference: str | None = Field(default=None, max_length=32)
    self_declared_made_for_kids: bool | None = None
    approved: bool = False


class BroadcastUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=100)
    scheduled_start_time: str | None = Field(default=None, max_length=64)
    description: str | None = Field(default=None, max_length=5000)
    scheduled_end_time: str | None = Field(default=None, max_length=64)
    privacy_status: str | None = Field(default=None, max_length=32)
    category_id: str | None = Field(default=None, max_length=16)
    enable_dvr: bool | None = None
    record_from_start: bool | None = None
    enable_auto_start: bool | None = None
    enable_auto_stop: bool | None = None
    latency_preference: str | None = Field(default=None, max_length=32)
    self_declared_made_for_kids: bool | None = None
    approved: bool = False


class BroadcastDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class BroadcastBindRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_id: str = Field(min_length=1, max_length=128)
    stream_id: str | None = Field(default=None, max_length=128)
    approved: bool = False


class BroadcastTransitionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    broadcast_id: str = Field(min_length=1, max_length=128)
    broadcast_status: str = Field(min_length=1, max_length=32)
    approved: bool = False


class StreamListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    stream_id: str | None = Field(default=None, max_length=128)
    max_results: int = Field(default=20, ge=1, le=50)
    page_token: str = Field(default="", max_length=2048)


class StreamGetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    stream_id: str = Field(min_length=1, max_length=128)


class StreamCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=100)
    ingestion_type: str = Field(default="rtmp", max_length=16)
    resolution: str = Field(default="720p", max_length=16)
    frame_rate: str = Field(default="30fps", max_length=16)
    description: str = Field(default="", max_length=5000)
    is_reusable: bool = True
    approved: bool = False


class StreamUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    stream_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=5000)
    approved: bool = False


class StreamDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    stream_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class LiveChatListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    live_chat_id: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=100, ge=1, le=2000)
    page_token: str = Field(default="", max_length=2048)


class LiveChatSendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    live_chat_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=10000)
    approved: bool = False


class LiveChatDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "integration": "youtube",
        "mode": "live",
        "oauth_connect_mode": "channel_manager",
        "capabilities": [
            "list and inspect upcoming, active, and completed live broadcasts",
            "create, update, bind, transition, and delete live broadcasts with explicit approval",
            "list and inspect live stream status and health",
            "create, update, and delete live streams with explicit approval",
            "list live chat messages",
            "send and delete live chat messages with explicit approval",
        ],
        "write_operations_require_approval": True,
        "secrets_exposed": False,
    }


@router.post("/connect")
async def connect(request: LiveConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "youtube_video_manager", request.redirect_uri.strip())
        result = build_youtube_manager_authorization(state, request.redirect_uri)
        capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), list) else []
        result["capabilities"] = [*capabilities, "manage YouTube live broadcasts and streams with explicit approval"]
        result["live_enabled"] = True
        return {**result, "state": state}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: LiveCallbackRequest) -> dict[str, object]:
    try:
        result = await exchange_youtube_manager_code(request.state, request.code)
        result["live_enabled"] = True
        return result
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"youtube live oauth failed: {exc}") from exc


def _route_error(exc: Exception, operation: str) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (httpx.HTTPError, RuntimeError)):
        return HTTPException(status_code=502, detail=f"youtube live {operation} failed: {exc}")
    return HTTPException(status_code=500, detail=f"youtube live {operation} failed")


@router.post("/broadcasts")
async def broadcasts(request: BroadcastListRequest) -> dict[str, object]:
    try:
        return await list_broadcasts(request.user_id, broadcast_status=request.broadcast_status, broadcast_id=request.broadcast_id, max_results=request.max_results, page_token=request.page_token)
    except Exception as exc:
        raise _route_error(exc, "broadcast list") from exc


@router.post("/broadcast")
async def broadcast(request: BroadcastGetRequest) -> dict[str, object]:
    try:
        return await get_broadcast(request.user_id, request.broadcast_id)
    except Exception as exc:
        raise _route_error(exc, "broadcast lookup") from exc


@router.post("/broadcast/create")
async def broadcast_create(request: BroadcastCreateRequest) -> dict[str, object]:
    try:
        return await create_broadcast(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "broadcast create") from exc


@router.post("/broadcast/update")
async def broadcast_update(request: BroadcastUpdateRequest) -> dict[str, object]:
    try:
        return await update_broadcast(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "broadcast update") from exc


@router.post("/broadcast/delete")
async def broadcast_delete(request: BroadcastDeleteRequest) -> dict[str, object]:
    try:
        return await delete_broadcast(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "broadcast delete") from exc


@router.post("/broadcast/bind")
async def broadcast_bind(request: BroadcastBindRequest) -> dict[str, object]:
    try:
        return await bind_broadcast(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "broadcast bind") from exc


@router.post("/broadcast/transition")
async def broadcast_transition(request: BroadcastTransitionRequest) -> dict[str, object]:
    try:
        return await transition_broadcast(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "broadcast transition") from exc


@router.post("/streams")
async def streams(request: StreamListRequest) -> dict[str, object]:
    try:
        return await list_streams(request.user_id, stream_id=request.stream_id, max_results=request.max_results, page_token=request.page_token)
    except Exception as exc:
        raise _route_error(exc, "stream list") from exc


@router.post("/stream")
async def stream(request: StreamGetRequest) -> dict[str, object]:
    try:
        return await get_stream(request.user_id, request.stream_id)
    except Exception as exc:
        raise _route_error(exc, "stream lookup") from exc


@router.post("/stream/create")
async def stream_create(request: StreamCreateRequest) -> dict[str, object]:
    try:
        return await create_stream(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "stream create") from exc


@router.post("/stream/update")
async def stream_update(request: StreamUpdateRequest) -> dict[str, object]:
    try:
        return await update_stream(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "stream update") from exc


@router.post("/stream/delete")
async def stream_delete(request: StreamDeleteRequest) -> dict[str, object]:
    try:
        return await delete_stream(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "stream delete") from exc


@router.post("/chat/messages")
async def chat_messages(request: LiveChatListRequest) -> dict[str, object]:
    try:
        return await list_live_chat_messages(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "live chat list") from exc


@router.post("/chat/send")
async def chat_send(request: LiveChatSendRequest) -> dict[str, object]:
    try:
        return await send_live_chat_message(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "live chat send") from exc


@router.post("/chat/delete")
async def chat_delete(request: LiveChatDeleteRequest) -> dict[str, object]:
    try:
        return await delete_live_chat_message(**request.model_dump())
    except Exception as exc:
        raise _route_error(exc, "live chat delete") from exc
