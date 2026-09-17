from __future__ import annotations

import base64
import re
from datetime import datetime
from uuid import uuid4

import httpx

from app.capabilities.store import get_integration_token
from app.capabilities.youtube import _auth_headers
from app.capabilities.youtube_video_manager import _manager_refresh, _manager_token

_API_BASE = "https://www.googleapis.com/youtube/v3"
_UPLOAD_CAPTIONS_URL = "https://www.googleapis.com/upload/youtube/v3/captions"
_MAX_RESULTS = 50
_SHORT_MAX_SECONDS = 180.0
_MAX_CAPTION_BYTES = 50 * 1024 * 1024
_ALLOWED_PRIVACY = {"private", "unlisted", "public"}
_ALLOWED_LICENSES = {"youtube", "creativeCommon"}
_ALLOWED_TRACK_KINDS = {"standard", "forced", "ASR"}
_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$"
)


def parse_iso_duration(value: str) -> float:
    value = value.strip().upper()
    match = _DURATION_RE.fullmatch(value)
    if not match:
        raise ValueError("duration must be an ISO 8601 duration")
    parts = {key: float(number or 0) for key, number in match.groupdict().items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def validate_short_eligibility(
    duration_seconds: float,
    width_pixels: int | None,
    height_pixels: int | None,
) -> dict[str, object]:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    if width_pixels is None or height_pixels is None:
        return {
            "eligible": False,
            "reason": "video dimensions are not available yet",
            "duration_seconds": duration_seconds,
            "aspect_ratio": None,
        }
    if width_pixels <= 0 or height_pixels <= 0:
        raise ValueError("video dimensions must be positive")
    aspect_ratio = width_pixels / height_pixels
    eligible = duration_seconds <= _SHORT_MAX_SECONDS and width_pixels <= height_pixels
    if duration_seconds > _SHORT_MAX_SECONDS:
        reason = "duration exceeds the three-minute Shorts limit"
    elif width_pixels > height_pixels:
        reason = "video is wider than tall; Shorts eligibility requires square or vertical media"
    else:
        reason = "eligible by duration and encoded orientation"
    return {
        "eligible": eligible,
        "reason": reason,
        "duration_seconds": duration_seconds,
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
        "aspect_ratio": round(aspect_ratio, 6),
    }


def _video_dimensions(video: dict[str, object]) -> tuple[int | None, int | None]:
    file_details = video.get("fileDetails")
    if not isinstance(file_details, dict):
        return None, None
    streams = file_details.get("videoStreams")
    if not isinstance(streams, list):
        return None, None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        width = stream.get("widthPixels")
        height = stream.get("heightPixels")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
    return None, None


def _video_duration(video: dict[str, object]) -> float:
    file_details = video.get("fileDetails")
    if isinstance(file_details, dict):
        duration_ms = file_details.get("durationMs")
        if isinstance(duration_ms, (int, float)):
            return float(duration_ms) / 1000.0
    content_details = video.get("contentDetails")
    if isinstance(content_details, dict):
        duration = content_details.get("duration")
        if isinstance(duration, str) and duration:
            return parse_iso_duration(duration)
    raise ValueError("video duration is not available")


def _short_evaluation(video: dict[str, object]) -> dict[str, object]:
    width, height = _video_dimensions(video)
    duration = _video_duration(video)
    result = validate_short_eligibility(duration, width, height)
    result["video_id"] = video.get("id")
    result["title"] = video.get("snippet", {}).get("title") if isinstance(video.get("snippet"), dict) else None
    result["processing_status"] = (
        video.get("processingDetails", {}).get("processingStatus")
        if isinstance(video.get("processingDetails"), dict)
        else None
    )
    return result


def _iso_datetime(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 datetime") from exc
    return value


def _validate_localizations(localizations: dict[str, dict[str, str]], default_language: str | None) -> dict[str, dict[str, str]]:
    if not isinstance(localizations, dict) or not localizations:
        raise ValueError("localizations must contain at least one language")
    if not default_language or not default_language.strip():
        raise ValueError("default_language is required when updating localizations")
    normalized: dict[str, dict[str, str]] = {}
    for language, values in localizations.items():
        key = str(language).strip()
        if not key or len(key) > 35:
            raise ValueError("localization language keys must be 1-35 characters")
        if not isinstance(values, dict):
            raise ValueError("each localization must be an object")
        title = str(values.get("title") or "").strip()
        description = str(values.get("description") or "")
        if not title or len(title) > 100:
            raise ValueError("localized title must be 1-100 characters")
        if len(description) > 5000:
            raise ValueError("localized description exceeds 5000 characters")
        normalized[key] = {"title": title, "description": description}
    return normalized


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
        if response.status_code == 204 or not getattr(response, "content", b""):
            return {}
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid advanced video response")
    return body


async def _upload_caption_media(
    user_id: str,
    method: str,
    params: dict[str, object],
    resource: dict[str, object],
    content: bytes,
    mime_type: str,
) -> dict[str, object]:
    boundary = f"indoone-{uuid4().hex}"
    metadata = __import__("json").dumps(resource, separators=(",", ":"))
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{metadata}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    token = await _manager_token(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": f"multipart/related; boundary={boundary}",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(
            method,
            _UPLOAD_CAPTIONS_URL,
            params={**params, "uploadType": "multipart"},
            headers=headers,
            content=body,
        )
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.request(
                method,
                _UPLOAD_CAPTIONS_URL,
                params={**params, "uploadType": "multipart"},
                headers=headers,
                content=body,
            )
        response.raise_for_status()
        result = response.json() if getattr(response, "content", b"") else {}
    if not isinstance(result, dict):
        raise RuntimeError("youtube returned an invalid caption upload response")
    return result


async def _download_caption_media(
    user_id: str,
    caption_id: str,
    *,
    tfmt: str | None = None,
    tlang: str | None = None,
) -> tuple[bytes, str]:
    token = await _manager_token(user_id)
    params: dict[str, object] = {}
    if tfmt and tfmt.strip():
        params["tfmt"] = tfmt.strip()
    if tlang and tlang.strip():
        params["tlang"] = tlang.strip()
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{_API_BASE}/captions/{caption_id.strip()}", params=params, headers=headers)
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            response = await client.get(
                f"{_API_BASE}/captions/{caption_id.strip()}",
                params=params,
                headers=_auth_headers(token),
            )
        response.raise_for_status()
    if len(response.content) > _MAX_CAPTION_BYTES:
        raise RuntimeError("downloaded caption exceeds backend size limit")
    return response.content, response.headers.get("content-type", "application/octet-stream")


async def get_video_advanced(user_id: str, video_id: str) -> dict[str, object]:
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id is required")
    result = await _request(
        user_id,
        "GET",
        "/videos",
        params={
            "part": "snippet,contentDetails,status,fileDetails,processingDetails,localizations,recordingDetails,paidProductPlacementDetails,liveStreamingDetails,suggestions",
            "id": video_id,
        },
    )
    items = result.get("items") if isinstance(result.get("items"), list) else []
    if not items:
        raise ValueError("video not found")
    video = items[0]
    if not isinstance(video, dict):
        raise RuntimeError("youtube returned an invalid video resource")
    return {
        "integration": "youtube",
        "operation": "get_video_advanced",
        "video": video,
        "shorts": _short_evaluation(video),
        "secrets_exposed": False,
    }


async def list_my_shorts(user_id: str, max_results: int = 20, page_token: str = "") -> dict[str, object]:
    if not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError("max_results must be between 1 and 50")
    channel = await _request(
        user_id,
        "GET",
        "/channels",
        params={"part": "contentDetails", "mine": "true"},
    )
    channels = channel.get("items") if isinstance(channel.get("items"), list) else []
    if not channels or not isinstance(channels[0], dict):
        return {"integration": "youtube", "shorts": [], "next_page_token": None, "secrets_exposed": False}
    details = channels[0].get("contentDetails")
    related = details.get("relatedPlaylists") if isinstance(details, dict) else None
    uploads_playlist_id = related.get("uploads") if isinstance(related, dict) else None
    if not isinstance(uploads_playlist_id, str) or not uploads_playlist_id:
        raise RuntimeError("authenticated channel has no uploads playlist")
    params: dict[str, object] = {
        "part": "contentDetails,snippet",
        "playlistId": uploads_playlist_id,
        "maxResults": max_results,
    }
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    playlist = await _request(user_id, "GET", "/playlistItems", params=params)
    playlist_items = playlist.get("items") if isinstance(playlist.get("items"), list) else []
    ids = [
        str(item.get("contentDetails", {}).get("videoId"))
        for item in playlist_items
        if isinstance(item, dict)
        and isinstance(item.get("contentDetails"), dict)
        and item.get("contentDetails", {}).get("videoId")
    ]
    if not ids:
        return {
            "integration": "youtube",
            "shorts": [],
            "next_page_token": playlist.get("nextPageToken"),
            "secrets_exposed": False,
        }
    videos_result = await _request(
        user_id,
        "GET",
        "/videos",
        params={"part": "snippet,contentDetails,fileDetails,processingDetails,status", "id": ",".join(ids)},
    )
    videos = videos_result.get("items") if isinstance(videos_result.get("items"), list) else []
    shorts = []
    for video in videos:
        if not isinstance(video, dict):
            continue
        evaluation = _short_evaluation(video)
        if evaluation["eligible"]:
            shorts.append(evaluation)
    return {
        "integration": "youtube",
        "shorts": shorts,
        "next_page_token": playlist.get("nextPageToken"),
        "page_info": playlist.get("pageInfo") if isinstance(playlist.get("pageInfo"), dict) else {},
        "secrets_exposed": False,
    }


async def update_video_advanced(
    user_id: str,
    video_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    category_id: str | None = None,
    tags: list[str] | None = None,
    default_language: str | None = None,
    privacy_status: str | None = None,
    embeddable: bool | None = None,
    license: str | None = None,
    public_stats_viewable: bool | None = None,
    publish_at: str | None = None,
    self_declared_made_for_kids: bool | None = None,
    contains_synthetic_media: bool | None = None,
    recording_date: str | None = None,
    localizations: dict[str, dict[str, str]] | None = None,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for advanced youtube video updates")
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id is required")
    if license is not None and license.strip() not in _ALLOWED_LICENSES:
        raise ValueError("license must be youtube or creativeCommon")
    if privacy_status is not None and privacy_status.strip() not in _ALLOWED_PRIVACY:
        raise ValueError("privacy_status must be private, unlisted, or public")
    existing_result = await get_video_advanced(user_id, video_id)
    existing = existing_result["video"]
    existing_snippet = existing.get("snippet") if isinstance(existing.get("snippet"), dict) else {}
    existing_status = existing.get("status") if isinstance(existing.get("status"), dict) else {}
    parts: list[str] = []
    resource: dict[str, object] = {"id": video_id}

    snippet_requested = any(value is not None for value in (title, description, category_id, tags, default_language)) or localizations is not None
    if snippet_requested:
        resolved_title = str(title if title is not None else existing_snippet.get("title") or "").strip()
        resolved_category = str(category_id if category_id is not None else existing_snippet.get("categoryId") or "").strip()
        if not resolved_title or not resolved_category:
            raise ValueError("title and category_id are required for snippet updates")
        snippet: dict[str, object] = {
            "title": resolved_title,
            "description": str(description if description is not None else existing_snippet.get("description") or ""),
            "categoryId": resolved_category,
        }
        if tags is not None:
            snippet["tags"] = [item.strip() for item in tags if item.strip()]
        elif isinstance(existing_snippet.get("tags"), list):
            snippet["tags"] = [str(item) for item in existing_snippet["tags"]]
        resolved_language = default_language if default_language is not None else existing_snippet.get("defaultLanguage")
        if resolved_language:
            snippet["defaultLanguage"] = str(resolved_language).strip()
        resource["snippet"] = snippet
        parts.append("snippet")

    status_requested = any(value is not None for value in (privacy_status, embeddable, license, public_stats_viewable, publish_at, self_declared_made_for_kids, contains_synthetic_media))
    if status_requested:
        status: dict[str, object] = {}
        for key, incoming in (
            ("privacyStatus", privacy_status),
            ("embeddable", embeddable),
            ("license", license),
            ("publicStatsViewable", public_stats_viewable),
            ("selfDeclaredMadeForKids", self_declared_made_for_kids),
            ("containsSyntheticMedia", contains_synthetic_media),
        ):
            if incoming is not None:
                status[key] = str(incoming).strip() if key in {"privacyStatus", "license"} else incoming
            elif key in existing_status:
                status[key] = existing_status[key]
        if publish_at is not None:
            normalized_publish_at = _iso_datetime(publish_at, "publish_at")
            target_privacy = str(status.get("privacyStatus") or "")
            if target_privacy != "private":
                raise ValueError("publish_at requires privacy_status=private")
            status["publishAt"] = normalized_publish_at
        resource["status"] = status
        parts.append("status")

    if recording_date is not None:
        resource["recordingDetails"] = {"recordingDate": _iso_datetime(recording_date, "recording_date")}
        parts.append("recordingDetails")

    if localizations is not None:
        default_language = str(default_language or existing_snippet.get("defaultLanguage") or "").strip()
        normalized_localizations = _validate_localizations(localizations, default_language)
        resource["localizations"] = normalized_localizations
        if "snippet" not in parts:
            raise RuntimeError("internal localization validation error")
        parts.append("localizations")

    if not parts:
        raise ValueError("at least one advanced video field must be provided")
    result = await _request(user_id, "PUT", "/videos", params={"part": ",".join(dict.fromkeys(parts))}, json=resource)
    return {
        "integration": "youtube",
        "operation": "update_video_advanced",
        "video": result,
        "secrets_exposed": False,
    }


async def list_captions(user_id: str, video_id: str) -> dict[str, object]:
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id is required")
    result = await _request(user_id, "GET", "/captions", params={"part": "id,snippet", "videoId": video_id})
    return {
        "integration": "youtube",
        "operation": "list_captions",
        "video_id": video_id,
        "captions": result.get("items") if isinstance(result.get("items"), list) else [],
        "secrets_exposed": False,
    }


def _decode_caption(content_base64: str) -> bytes:
    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("content_base64 is invalid") from exc
    if not content:
        raise ValueError("caption content is required")
    if len(content) > _MAX_CAPTION_BYTES:
        raise ValueError("caption file exceeds the backend 50MB limit")
    return content


async def insert_caption(
    user_id: str,
    video_id: str,
    language: str,
    name: str,
    content_base64: str,
    mime_type: str = "application/octet-stream",
    *,
    track_kind: str = "standard",
    is_draft: bool = False,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for caption insertion")
    video_id = video_id.strip()
    language = language.strip()
    name = name.strip()
    track_kind = track_kind.strip()
    mime_type = mime_type.strip() or "application/octet-stream"
    if not video_id or not language:
        raise ValueError("video_id and language are required")
    if not name or len(name) > 150:
        raise ValueError("caption name must be 1-150 characters")
    if track_kind not in _ALLOWED_TRACK_KINDS:
        raise ValueError("track_kind must be standard, forced, or ASR")
    content = _decode_caption(content_base64)
    resource = {
        "snippet": {
            "videoId": video_id,
            "language": language,
            "name": name,
            "trackKind": track_kind,
            "isDraft": is_draft,
        }
    }
    result = await _upload_caption_media(user_id, "POST", {"part": "snippet"}, resource, content, mime_type)
    return {"integration": "youtube", "operation": "insert_caption", "caption": result, "secrets_exposed": False}


async def update_caption(
    user_id: str,
    caption_id: str,
    *,
    content_base64: str | None = None,
    mime_type: str = "application/octet-stream",
    is_draft: bool | None = None,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for caption updates")
    caption_id = caption_id.strip()
    if not caption_id:
        raise ValueError("caption_id is required")
    if content_base64 is None and is_draft is None:
        raise ValueError("content_base64 or is_draft must be provided")
    resource: dict[str, object] = {"id": caption_id}
    content = b""
    if is_draft is not None:
        resource["snippet"] = {"isDraft": is_draft}
    if content_base64 is not None:
        content = _decode_caption(content_base64)
    if content:
        result = await _upload_caption_media(
            user_id,
            "PUT",
            {"part": "id,snippet" if "snippet" in resource else "id"},
            resource,
            content,
            mime_type.strip() or "application/octet-stream",
        )
    else:
        result = await _request(user_id, "PUT", "/captions", params={"part": "snippet"}, json=resource)
    return {"integration": "youtube", "operation": "update_caption", "caption": result, "secrets_exposed": False}


async def delete_caption(user_id: str, caption_id: str, *, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for caption deletion")
    caption_id = caption_id.strip()
    if not caption_id:
        raise ValueError("caption_id is required")
    await _request(user_id, "DELETE", "/captions", params={"id": caption_id})
    return {"integration": "youtube", "operation": "delete_caption", "caption_id": caption_id, "deleted": True, "secrets_exposed": False}


async def download_caption(
    user_id: str,
    caption_id: str,
    *,
    tfmt: str | None = None,
    tlang: str | None = None,
) -> dict[str, object]:
    caption_id = caption_id.strip()
    if not caption_id:
        raise ValueError("caption_id is required")
    content, mime_type = await _download_caption_media(user_id, caption_id, tfmt=tfmt, tlang=tlang)
    return {
        "integration": "youtube",
        "operation": "download_caption",
        "caption_id": caption_id,
        "mime_type": mime_type,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "secrets_exposed": False,
    }
