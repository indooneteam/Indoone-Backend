from __future__ import annotations

from datetime import datetime

import httpx

from app.capabilities.store import get_integration_token
from app.capabilities.youtube import _auth_headers
from app.capabilities.youtube_video_manager import _manager_refresh, _manager_scope, _manager_token

_API_BASE = "https://www.googleapis.com/youtube/v3"
_BROADCAST_STATUSES = {"upcoming", "active", "completed", "all"}
_TRANSITIONS = {"testing", "live", "complete"}
_PRIVACY = {"public", "private", "unlisted"}
_LATENCY = {"normal", "low", "ultraLow"}
_STREAM_INGESTION = {"rtmp", "rtmps", "dash", "hls"}
_STREAM_RESOLUTIONS = {"2160p", "1440p", "1080p", "720p", "480p", "360p", "240p"}
_STREAM_FRAME_RATES = {"30fps", "60fps"}
_MAX_TITLE = 100
_MAX_DESCRIPTION = 5000
_MAX_CHAT_TEXT = 10000


def _require_approval(approved: bool, operation: str) -> None:
    if not approved:
        raise PermissionError(f"explicit approval is required for youtube live {operation}")


def _clean_required(value: str, field_name: str, max_length: int | None = None) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters")
    return normalized


def _validate_datetime(value: str, field_name: str) -> str:
    normalized = _clean_required(value, field_name)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return normalized


def _validate_privacy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _PRIVACY:
        raise ValueError("privacy_status must be public, private, or unlisted")
    return normalized


def _request_body_or_empty(response: httpx.Response) -> dict[str, object]:
    if response.status_code == 204 or not getattr(response, "content", b""):
        return {}
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid live api response")
    return body


async def _request(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    json: dict[str, object] | None = None,
    timeout: float = 30.0,
) -> dict[str, object]:
    token = await _manager_token(user_id)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            f"{_API_BASE}{path}",
            params=params,
            headers=_auth_headers(token),
            json=json,
        )
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            response = await client.request(
                method,
                f"{_API_BASE}{path}",
                params=params,
                headers=_auth_headers(token),
                json=json,
            )
        response.raise_for_status()
        return _request_body_or_empty(response)


def _ensure_live_scope(user_id: str) -> None:
    _manager_scope(user_id)


def _normalize_page(max_results: int, page_token: str) -> tuple[int, str]:
    if not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50")
    return max_results, page_token.strip()


async def list_broadcasts(
    user_id: str,
    *,
    broadcast_status: str | None = None,
    broadcast_id: str | None = None,
    max_results: int = 20,
    page_token: str = "",
) -> dict[str, object]:
    _ensure_live_scope(user_id)
    max_results, page_token = _normalize_page(max_results, page_token)
    requested_id = (broadcast_id or "").strip()
    requested_status = (broadcast_status or "").strip().lower()
    if requested_id and requested_status:
        raise ValueError("broadcast_id and broadcast_status cannot be used together")
    params: dict[str, object] = {
        "part": "id,snippet,contentDetails,status",
        "maxResults": max_results,
    }
    if requested_id:
        params["id"] = requested_id
    else:
        params["mine"] = "true"
        params["broadcastType"] = "all"
        if requested_status:
            if requested_status not in _BROADCAST_STATUSES:
                raise ValueError("broadcast_status must be upcoming, active, completed, or all")
            params["broadcastStatus"] = requested_status
    if page_token:
        params["pageToken"] = page_token
    result = await _request(user_id, "GET", "/liveBroadcasts", params=params)
    return {
        "integration": "youtube",
        "operation": "list_broadcasts",
        "broadcasts": result.get("items") if isinstance(result.get("items"), list) else [],
        "next_page_token": result.get("nextPageToken", ""),
        "page_info": result.get("pageInfo") if isinstance(result.get("pageInfo"), dict) else {},
        "secrets_exposed": False,
    }


async def get_broadcast(user_id: str, broadcast_id: str) -> dict[str, object]:
    broadcast_id = _clean_required(broadcast_id, "broadcast_id")
    result = await list_broadcasts(user_id, broadcast_id=broadcast_id, max_results=1)
    broadcasts = result.get("broadcasts") if isinstance(result.get("broadcasts"), list) else []
    if not broadcasts:
        raise ValueError("broadcast not found")
    return {
        "integration": "youtube",
        "operation": "get_broadcast",
        "broadcast": broadcasts[0],
        "secrets_exposed": False,
    }


def build_broadcast_resource(
    *,
    title: str,
    scheduled_start_time: str,
    description: str = "",
    scheduled_end_time: str | None = None,
    privacy_status: str = "private",
    category_id: str | None = None,
    enable_dvr: bool | None = None,
    record_from_start: bool | None = None,
    enable_auto_start: bool | None = None,
    enable_auto_stop: bool | None = None,
    latency_preference: str | None = None,
    self_declared_made_for_kids: bool | None = None,
) -> dict[str, object]:
    resolved_title = _clean_required(title, "title", _MAX_TITLE)
    resolved_start = _validate_datetime(scheduled_start_time, "scheduled_start_time")
    resolved_description = description.strip()
    if len(resolved_description) > _MAX_DESCRIPTION:
        raise ValueError("description exceeds 5000 characters")
    resource: dict[str, object] = {
        "snippet": {
            "title": resolved_title,
            "scheduledStartTime": resolved_start,
        },
        "status": {"privacyStatus": _validate_privacy(privacy_status)},
    }
    snippet = resource["snippet"]
    if not isinstance(snippet, dict):
        raise RuntimeError("internal broadcast payload error")
    if resolved_description:
        snippet["description"] = resolved_description
    if category_id is not None:
        snippet["categoryId"] = _clean_required(category_id, "category_id", 16)
    if scheduled_end_time is not None:
        snippet["scheduledEndTime"] = _validate_datetime(scheduled_end_time, "scheduled_end_time")
    content: dict[str, object] = {}
    optional_content = {
        "enableDvr": enable_dvr,
        "recordFromStart": record_from_start,
        "enableAutoStart": enable_auto_start,
        "enableAutoStop": enable_auto_stop,
    }
    for key, value in optional_content.items():
        if value is not None:
            content[key] = value
    if latency_preference is not None:
        latency = latency_preference.strip()
        if latency not in _LATENCY:
            raise ValueError("latency_preference must be normal, low, or ultraLow")
        content["latencyPreference"] = latency
    if content:
        resource["contentDetails"] = content
    if self_declared_made_for_kids is not None:
        status = resource["status"]
        if isinstance(status, dict):
            status["selfDeclaredMadeForKids"] = self_declared_made_for_kids
    return resource


async def create_broadcast(
    user_id: str,
    *,
    title: str,
    scheduled_start_time: str,
    description: str = "",
    scheduled_end_time: str | None = None,
    privacy_status: str = "private",
    category_id: str | None = None,
    enable_dvr: bool | None = None,
    record_from_start: bool | None = None,
    enable_auto_start: bool | None = None,
    enable_auto_stop: bool | None = None,
    latency_preference: str | None = None,
    self_declared_made_for_kids: bool | None = None,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "broadcast creation")
    payload = build_broadcast_resource(
        title=title,
        scheduled_start_time=scheduled_start_time,
        description=description,
        scheduled_end_time=scheduled_end_time,
        privacy_status=privacy_status,
        category_id=category_id,
        enable_dvr=enable_dvr,
        record_from_start=record_from_start,
        enable_auto_start=enable_auto_start,
        enable_auto_stop=enable_auto_stop,
        latency_preference=latency_preference,
        self_declared_made_for_kids=self_declared_made_for_kids,
    )
    result = await _request(
        user_id,
        "POST",
        "/liveBroadcasts",
        params={"part": "snippet,status,contentDetails"},
        json=payload,
    )
    return {
        "integration": "youtube",
        "operation": "create_broadcast",
        "broadcast": result,
        "secrets_exposed": False,
    }


async def update_broadcast(
    user_id: str,
    broadcast_id: str,
    *,
    title: str | None = None,
    scheduled_start_time: str | None = None,
    description: str | None = None,
    scheduled_end_time: str | None = None,
    privacy_status: str | None = None,
    category_id: str | None = None,
    enable_dvr: bool | None = None,
    record_from_start: bool | None = None,
    enable_auto_start: bool | None = None,
    enable_auto_stop: bool | None = None,
    latency_preference: str | None = None,
    self_declared_made_for_kids: bool | None = None,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "broadcast update")
    broadcast_id = _clean_required(broadcast_id, "broadcast_id")
    if all(value is None for value in (
        title, scheduled_start_time, description, scheduled_end_time, privacy_status,
        category_id, enable_dvr, record_from_start, enable_auto_start,
        enable_auto_stop, latency_preference, self_declared_made_for_kids,
    )):
        raise ValueError("at least one broadcast field must be provided")
    existing = await get_broadcast(user_id, broadcast_id)
    current = existing["broadcast"] if isinstance(existing.get("broadcast"), dict) else {}
    current_snippet = current.get("snippet") if isinstance(current.get("snippet"), dict) else {}
    current_status = current.get("status") if isinstance(current.get("status"), dict) else {}
    current_content = current.get("contentDetails") if isinstance(current.get("contentDetails"), dict) else {}
    resolved_start = str(scheduled_start_time or current_snippet.get("scheduledStartTime") or "")
    resolved_title = str(title if title is not None else current_snippet.get("title") or "")
    if not resolved_title or not resolved_start:
        raise ValueError("existing broadcast is missing required snippet fields")
    resolved_end = scheduled_end_time if scheduled_end_time is not None else current_snippet.get("scheduledEndTime")
    payload = build_broadcast_resource(
        title=resolved_title,
        scheduled_start_time=resolved_start,
        description=str(description if description is not None else current_snippet.get("description") or ""),
        scheduled_end_time=str(resolved_end) if resolved_end else None,
        privacy_status=str(privacy_status if privacy_status is not None else current_status.get("privacyStatus") or "private"),
        category_id=str(category_id if category_id is not None else current_snippet.get("categoryId") or "22"),
        enable_dvr=enable_dvr if enable_dvr is not None else current_content.get("enableDvr"),
        record_from_start=record_from_start if record_from_start is not None else current_content.get("recordFromStart"),
        enable_auto_start=enable_auto_start if enable_auto_start is not None else current_content.get("enableAutoStart"),
        enable_auto_stop=enable_auto_stop if enable_auto_stop is not None else current_content.get("enableAutoStop"),
        latency_preference=latency_preference if latency_preference is not None else current_content.get("latencyPreference"),
        self_declared_made_for_kids=(
            self_declared_made_for_kids
            if self_declared_made_for_kids is not None
            else current_status.get("selfDeclaredMadeForKids")
        ),
    )
    payload["id"] = broadcast_id
    parts = ["snippet", "status"]
    if "contentDetails" in payload:
        parts.append("contentDetails")
    result = await _request(
        user_id,
        "PUT",
        "/liveBroadcasts",
        params={"part": ",".join(parts)},
        json=payload,
    )
    return {
        "integration": "youtube",
        "operation": "update_broadcast",
        "broadcast": result,
        "secrets_exposed": False,
    }


async def delete_broadcast(user_id: str, broadcast_id: str, *, approved: bool = False) -> dict[str, object]:
    _require_approval(approved, "broadcast deletion")
    broadcast_id = _clean_required(broadcast_id, "broadcast_id")
    await _request(user_id, "DELETE", "/liveBroadcasts", params={"id": broadcast_id})
    return {
        "integration": "youtube",
        "operation": "delete_broadcast",
        "broadcast_id": broadcast_id,
        "deleted": True,
        "secrets_exposed": False,
    }


async def bind_broadcast(
    user_id: str,
    broadcast_id: str,
    stream_id: str | None = None,
    *,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "broadcast binding")
    broadcast_id = _clean_required(broadcast_id, "broadcast_id")
    params: dict[str, object] = {"id": broadcast_id, "part": "id,snippet"}
    operation = "unbind_broadcast" if not (stream_id or "").strip() else "bind_broadcast"
    if stream_id and stream_id.strip():
        params["streamId"] = stream_id.strip()
    result = await _request(user_id, "POST", "/liveBroadcasts/bind", params=params)
    return {
        "integration": "youtube",
        "operation": operation,
        "broadcast": result,
        "secrets_exposed": False,
    }


async def transition_broadcast(
    user_id: str,
    broadcast_id: str,
    broadcast_status: str,
    *,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "broadcast transition")
    broadcast_id = _clean_required(broadcast_id, "broadcast_id")
    normalized_status = broadcast_status.strip()
    if normalized_status not in _TRANSITIONS:
        raise ValueError("broadcast_status must be testing, live, or complete")
    result = await _request(
        user_id,
        "POST",
        "/liveBroadcasts/transition",
        params={"broadcastStatus": normalized_status, "id": broadcast_id, "part": "id,snippet,status,contentDetails"},
    )
    return {
        "integration": "youtube",
        "operation": "transition_broadcast",
        "broadcast": result,
        "secrets_exposed": False,
    }


async def list_streams(
    user_id: str,
    *,
    stream_id: str | None = None,
    max_results: int = 20,
    page_token: str = "",
) -> dict[str, object]:
    _ensure_live_scope(user_id)
    max_results, page_token = _normalize_page(max_results, page_token)
    requested_id = (stream_id or "").strip()
    params: dict[str, object] = {
        "part": "id,snippet,cdn,status",
        "maxResults": max_results,
    }
    if requested_id:
        params["id"] = requested_id
    else:
        params["mine"] = "true"
    if page_token:
        params["pageToken"] = page_token
    result = await _request(user_id, "GET", "/liveStreams", params=params)
    return {
        "integration": "youtube",
        "operation": "list_streams",
        "streams": result.get("items") if isinstance(result.get("items"), list) else [],
        "next_page_token": result.get("nextPageToken", ""),
        "page_info": result.get("pageInfo") if isinstance(result.get("pageInfo"), dict) else {},
        "secrets_exposed": False,
    }


async def get_stream(user_id: str, stream_id: str) -> dict[str, object]:
    stream_id = _clean_required(stream_id, "stream_id")
    result = await list_streams(user_id, stream_id=stream_id, max_results=1)
    streams = result.get("streams") if isinstance(result.get("streams"), list) else []
    if not streams:
        raise ValueError("stream not found")
    return {
        "integration": "youtube",
        "operation": "get_stream",
        "stream": streams[0],
        "secrets_exposed": False,
    }


def build_stream_resource(
    *,
    title: str,
    ingestion_type: str = "rtmp",
    resolution: str = "720p",
    frame_rate: str = "30fps",
    description: str = "",
    is_reusable: bool = True,
) -> dict[str, object]:
    resolved_title = _clean_required(title, "title", _MAX_TITLE)
    ingestion = ingestion_type.strip().lower()
    if ingestion not in _STREAM_INGESTION:
        raise ValueError("ingestion_type must be rtmp, rtmps, dash, or hls")
    resolved_resolution = resolution.strip()
    if resolved_resolution not in _STREAM_RESOLUTIONS:
        raise ValueError("resolution must be 2160p, 1440p, 1080p, 720p, 480p, 360p, or 240p")
    resolved_frame_rate = frame_rate.strip()
    if resolved_frame_rate not in _STREAM_FRAME_RATES:
        raise ValueError("frame_rate must be 30fps or 60fps")
    resolved_description = description.strip()
    if len(resolved_description) > _MAX_DESCRIPTION:
        raise ValueError("description exceeds 5000 characters")
    snippet: dict[str, object] = {"title": resolved_title}
    if resolved_description:
        snippet["description"] = resolved_description
    return {
        "snippet": snippet,
        "cdn": {
            "ingestionType": ingestion,
            "resolution": resolved_resolution,
            "frameRate": resolved_frame_rate,
        },
        "contentDetails": {"isReusable": bool(is_reusable)},
    }


async def create_stream(
    user_id: str,
    *,
    title: str,
    ingestion_type: str = "rtmp",
    resolution: str = "720p",
    frame_rate: str = "30fps",
    description: str = "",
    is_reusable: bool = True,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "stream creation")
    payload = build_stream_resource(
        title=title,
        ingestion_type=ingestion_type,
        resolution=resolution,
        frame_rate=frame_rate,
        description=description,
        is_reusable=is_reusable,
    )
    result = await _request(
        user_id,
        "POST",
        "/liveStreams",
        params={"part": "snippet,cdn,contentDetails"},
        json=payload,
    )
    return {
        "integration": "youtube",
        "operation": "create_stream",
        "stream": result,
        "secrets_exposed": False,
    }


async def update_stream(
    user_id: str,
    stream_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "stream update")
    stream_id = _clean_required(stream_id, "stream_id")
    if title is None and description is None:
        raise ValueError("at least one of title or description must be provided")
    existing = await get_stream(user_id, stream_id)
    current = existing["stream"] if isinstance(existing.get("stream"), dict) else {}
    current_snippet = current.get("snippet") if isinstance(current.get("snippet"), dict) else {}
    resolved_title = _clean_required(str(title if title is not None else current_snippet.get("title") or ""), "title", _MAX_TITLE)
    resolved_description = str(description if description is not None else current_snippet.get("description") or "")
    if len(resolved_description.strip()) > _MAX_DESCRIPTION:
        raise ValueError("description exceeds 5000 characters")
    payload: dict[str, object] = {
        "id": stream_id,
        "snippet": {"title": resolved_title, "description": resolved_description.strip()},
    }
    result = await _request(
        user_id,
        "PUT",
        "/liveStreams",
        params={"part": "snippet"},
        json=payload,
    )
    return {
        "integration": "youtube",
        "operation": "update_stream",
        "stream": result,
        "secrets_exposed": False,
    }


async def delete_stream(user_id: str, stream_id: str, *, approved: bool = False) -> dict[str, object]:
    _require_approval(approved, "stream deletion")
    stream_id = _clean_required(stream_id, "stream_id")
    await _request(user_id, "DELETE", "/liveStreams", params={"id": stream_id})
    return {
        "integration": "youtube",
        "operation": "delete_stream",
        "stream_id": stream_id,
        "deleted": True,
        "secrets_exposed": False,
    }


async def list_live_chat_messages(
    user_id: str,
    live_chat_id: str,
    *,
    max_results: int = 100,
    page_token: str = "",
) -> dict[str, object]:
    live_chat_id = _clean_required(live_chat_id, "live_chat_id")
    if not 1 <= max_results <= 2000:
        raise ValueError("max_results must be between 1 and 2000")
    params: dict[str, object] = {
        "part": "id,snippet,authorDetails",
        "liveChatId": live_chat_id,
        "maxResults": max_results,
    }
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    result = await _request(user_id, "GET", "/liveChat/messages", params=params)
    return {
        "integration": "youtube",
        "operation": "list_live_chat_messages",
        "live_chat_id": live_chat_id,
        "messages": result.get("items") if isinstance(result.get("items"), list) else [],
        "next_page_token": result.get("nextPageToken", ""),
        "polling_interval_millis": result.get("pollingIntervalMillis"),
        "offline_at": result.get("offlineAt"),
        "page_info": result.get("pageInfo") if isinstance(result.get("pageInfo"), dict) else {},
        "active_poll_item": result.get("activePollItem"),
        "secrets_exposed": False,
    }


async def send_live_chat_message(
    user_id: str,
    live_chat_id: str,
    text: str,
    *,
    approved: bool = False,
) -> dict[str, object]:
    _require_approval(approved, "live chat message")
    live_chat_id = _clean_required(live_chat_id, "live_chat_id")
    message = _clean_required(text, "text", _MAX_CHAT_TEXT)
    result = await _request(
        user_id,
        "POST",
        "/liveChat/messages",
        params={"part": "snippet"},
        json={
            "snippet": {
                "liveChatId": live_chat_id,
                "type": "textMessageEvent",
                "textMessageDetails": {"messageText": message},
            }
        },
    )
    return {
        "integration": "youtube",
        "operation": "send_live_chat_message",
        "message": result,
        "secrets_exposed": False,
    }


async def delete_live_chat_message(user_id: str, message_id: str, *, approved: bool = False) -> dict[str, object]:
    _require_approval(approved, "live chat message deletion")
    message_id = _clean_required(message_id, "message_id")
    await _request(user_id, "DELETE", "/liveChat/messages", params={"id": message_id})
    return {
        "integration": "youtube",
        "operation": "delete_live_chat_message",
        "message_id": message_id,
        "deleted": True,
        "secrets_exposed": False,
    }
