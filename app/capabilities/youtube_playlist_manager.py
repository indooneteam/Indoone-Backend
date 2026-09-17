from __future__ import annotations

import httpx

from app.capabilities.store import get_integration_token
from app.capabilities.youtube_video_manager import _manager_refresh, _manager_token

_API_BASE = "https://www.googleapis.com/youtube/v3"
_MAX_RESULTS = 50
_PRIVACY_STATUSES = {"private", "unlisted", "public"}


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _validate_results(max_results: int) -> None:
    if not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError("max_results must be between 1 and 50")


def _validate_playlist_fields(
    *, title: str | None = None, description: str | None = None, privacy_status: str | None = None
) -> tuple[str | None, str | None, str | None]:
    normalized_title = title.strip() if title is not None else None
    normalized_description = description if description is not None else None
    normalized_privacy = privacy_status.strip().lower() if privacy_status is not None else None
    if normalized_title is not None and not normalized_title:
        raise ValueError("title is required")
    if normalized_title is not None and len(normalized_title) > 150:
        raise ValueError("title must be 150 characters or fewer")
    if normalized_description is not None and len(normalized_description) > 5000:
        raise ValueError("description must be 5000 characters or fewer")
    if normalized_privacy is not None and normalized_privacy not in _PRIVACY_STATUSES:
        raise ValueError("privacy_status must be private, unlisted, or public")
    return normalized_title, normalized_description, normalized_privacy


async def _request(user_id: str, method: str, path: str, **kwargs: object) -> dict[str, object] | None:
    token = await _manager_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, f"{_API_BASE}{path}", headers=_auth_headers(token), **kwargs)
        if response.status_code == 401:
            row = get_integration_token(user_id.strip(), "youtube") or {}
            token = await _manager_refresh(user_id, row)
            response = await client.request(method, f"{_API_BASE}{path}", headers=_auth_headers(token), **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid response")
    return body


async def list_playlists(user_id: str, max_results: int = 20, page_token: str = "") -> dict[str, object]:
    _validate_results(max_results)
    params: dict[str, object] = {"part": "snippet,contentDetails,status", "mine": "true", "maxResults": max_results}
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    body = await _request(user_id, "GET", "/playlists", params=params)
    body = body or {}
    return {
        "integration": "youtube",
        "playlists": body.get("items") if isinstance(body.get("items"), list) else [],
        "next_page_token": body.get("nextPageToken"),
        "total_results": body.get("pageInfo", {}).get("totalResults", 0) if isinstance(body.get("pageInfo"), dict) else 0,
        "secrets_exposed": False,
    }


async def get_playlist(user_id: str, playlist_id: str) -> dict[str, object]:
    playlist_id = playlist_id.strip()
    if not playlist_id:
        raise ValueError("playlist_id is required")
    body = await _request(user_id, "GET", "/playlists", params={"part": "snippet,contentDetails,status", "id": playlist_id})
    items = body.get("items") if isinstance(body, dict) and isinstance(body.get("items"), list) else []
    if not items:
        raise ValueError("playlist not found")
    return {"integration": "youtube", "playlist": items[0], "secrets_exposed": False}


async def create_playlist(
    user_id: str, title: str, description: str = "", privacy_status: str = "private", *, approved: bool = False
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist create operations")
    normalized_title, normalized_description, normalized_privacy = _validate_playlist_fields(
        title=title, description=description, privacy_status=privacy_status
    )
    body = await _request(
        user_id,
        "POST",
        "/playlists",
        params={"part": "snippet,status"},
        json={
            "snippet": {"title": normalized_title, "description": normalized_description or ""},
            "status": {"privacyStatus": normalized_privacy or "private"},
        },
    )
    return {"integration": "youtube", "operation": "create_playlist", "playlist": body or {}, "secrets_exposed": False}


async def update_playlist(
    user_id: str,
    playlist_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    privacy_status: str | None = None,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist update operations")
    playlist_id = playlist_id.strip()
    if not playlist_id:
        raise ValueError("playlist_id is required")
    if title is None and description is None and privacy_status is None:
        raise ValueError("at least one playlist field must be provided")
    existing = (await get_playlist(user_id, playlist_id)).get("playlist")
    current = existing if isinstance(existing, dict) else {}
    current_snippet = current.get("snippet") if isinstance(current.get("snippet"), dict) else {}
    current_status = current.get("status") if isinstance(current.get("status"), dict) else {}
    normalized_title, normalized_description, normalized_privacy = _validate_playlist_fields(
        title=title, description=description, privacy_status=privacy_status
    )
    resource: dict[str, object] = {"id": playlist_id}
    parts: list[str] = []
    if title is not None or description is not None:
        resolved_title = normalized_title if normalized_title is not None else str(current_snippet.get("title") or "").strip()
        if not resolved_title:
            raise ValueError("title is required for snippet updates")
        resource["snippet"] = {
            "title": resolved_title,
            "description": normalized_description if normalized_description is not None else str(current_snippet.get("description") or ""),
        }
        parts.append("snippet")
    if privacy_status is not None:
        resource["status"] = {"privacyStatus": normalized_privacy or str(current_status.get("privacyStatus") or "private")}
        parts.append("status")
    body = await _request(user_id, "PUT", "/playlists", params={"part": ",".join(parts)}, json=resource)
    return {"integration": "youtube", "operation": "update_playlist", "playlist": body or {}, "secrets_exposed": False}


async def delete_playlist(user_id: str, playlist_id: str, *, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist delete operations")
    playlist_id = playlist_id.strip()
    if not playlist_id:
        raise ValueError("playlist_id is required")
    await _request(user_id, "DELETE", "/playlists", params={"id": playlist_id})
    return {"integration": "youtube", "operation": "delete_playlist", "playlist_id": playlist_id, "deleted": True, "secrets_exposed": False}


async def list_playlist_items(user_id: str, playlist_id: str, max_results: int = 50, page_token: str = "") -> dict[str, object]:
    _validate_results(max_results)
    playlist_id = playlist_id.strip()
    if not playlist_id:
        raise ValueError("playlist_id is required")
    params: dict[str, object] = {"part": "snippet,contentDetails,status", "playlistId": playlist_id, "maxResults": max_results}
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    body = await _request(user_id, "GET", "/playlistItems", params=params)
    body = body or {}
    return {
        "integration": "youtube",
        "playlist_id": playlist_id,
        "items": body.get("items") if isinstance(body.get("items"), list) else [],
        "next_page_token": body.get("nextPageToken"),
        "total_results": body.get("pageInfo", {}).get("totalResults", 0) if isinstance(body.get("pageInfo"), dict) else 0,
        "secrets_exposed": False,
    }


async def add_video_to_playlist(
    user_id: str, playlist_id: str, video_id: str, position: int | None = None, *, approved: bool = False
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist add operations")
    playlist_id, video_id = playlist_id.strip(), video_id.strip()
    if not playlist_id:
        raise ValueError("playlist_id is required")
    if not video_id:
        raise ValueError("video_id is required")
    if position is not None and position < 0:
        raise ValueError("position must be zero or greater")
    snippet: dict[str, object] = {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}
    if position is not None:
        snippet["position"] = position
    body = await _request(user_id, "POST", "/playlistItems", params={"part": "snippet"}, json={"snippet": snippet})
    return {"integration": "youtube", "operation": "add_video_to_playlist", "item": body or {}, "secrets_exposed": False}


async def remove_playlist_item(user_id: str, playlist_item_id: str, *, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist remove operations")
    playlist_item_id = playlist_item_id.strip()
    if not playlist_item_id:
        raise ValueError("playlist_item_id is required")
    await _request(user_id, "DELETE", "/playlistItems", params={"id": playlist_item_id})
    return {"integration": "youtube", "operation": "remove_playlist_item", "playlist_item_id": playlist_item_id, "deleted": True, "secrets_exposed": False}


async def reorder_playlist_item(
    user_id: str, playlist_item_id: str, position: int, *, approved: bool = False
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube playlist reorder operations")
    playlist_item_id = playlist_item_id.strip()
    if not playlist_item_id:
        raise ValueError("playlist_item_id is required")
    if position < 0:
        raise ValueError("position must be zero or greater")
    lookup = await _request(user_id, "GET", "/playlistItems", params={"part": "snippet,contentDetails", "id": playlist_item_id})
    items = lookup.get("items") if isinstance(lookup, dict) and isinstance(lookup.get("items"), list) else []
    if not items or not isinstance(items[0], dict):
        raise ValueError("playlist item not found")
    current = items[0]
    snippet = current.get("snippet") if isinstance(current.get("snippet"), dict) else {}
    content = current.get("contentDetails") if isinstance(current.get("contentDetails"), dict) else {}
    playlist_id = str(snippet.get("playlistId") or "").strip()
    video_id = str(content.get("videoId") or "").strip()
    if not playlist_id or not video_id:
        raise RuntimeError("playlist item is missing required resource details")
    payload = {
        "id": playlist_item_id,
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {"kind": "youtube#video", "videoId": video_id},
            "position": position,
        },
    }
    body = await _request(user_id, "PUT", "/playlistItems", params={"part": "snippet"}, json=payload)
    return {"integration": "youtube", "operation": "reorder_playlist_item", "item": body or {}, "secrets_exposed": False}
