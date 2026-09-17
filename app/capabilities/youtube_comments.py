from __future__ import annotations

import httpx

from app.capabilities.store import get_integration_token
from app.capabilities.youtube import _auth_headers
from app.capabilities.youtube_video_manager import _manager_refresh, _manager_token

_API_BASE = "https://www.googleapis.com/youtube/v3"
_MAX_RESULTS = 100
_MODERATION_STATUSES = {"heldForReview", "published", "rejected"}
_ORDERS = {"time", "relevance"}
_TEXT_FORMATS = {"html", "plainText"}
_MAX_COMMENT_CHARS = 10_000


def _validate_comment_text(text: str) -> str:
    value = text.strip()
    if not value:
        raise ValueError("comment text is required")
    if len(value) > _MAX_COMMENT_CHARS:
        raise ValueError("comment text exceeds 10000 characters")
    return value


def _validate_optional_filter(value: str | None, allowed: set[str], field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized and normalized not in allowed:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(allowed))}")
    return normalized or None


def _build_thread_list_params(
    *,
    video_id: str | None = None,
    channel_id: str | None = None,
    all_threads_related_to_channel_id: str | None = None,
    max_results: int = 20,
    page_token: str = "",
    moderation_status: str | None = None,
    search_terms: str | None = None,
    order: str | None = None,
    text_format: str = "plainText",
) -> dict[str, object]:
    filters = [
        bool(video_id and video_id.strip()),
        bool(channel_id and channel_id.strip()),
        bool(all_threads_related_to_channel_id and all_threads_related_to_channel_id.strip()),
    ]
    if sum(filters) != 1:
        raise ValueError("exactly one of video_id, channel_id, or all_threads_related_to_channel_id is required")
    if not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError("max_results must be between 1 and 100")
    params: dict[str, object] = {
        "part": "snippet,replies",
        "maxResults": max_results,
        "textFormat": _validate_optional_filter(text_format, _TEXT_FORMATS, "text_format") or "plainText",
    }
    if video_id and video_id.strip():
        params["videoId"] = video_id.strip()
    elif channel_id and channel_id.strip():
        params["channelId"] = channel_id.strip()
    else:
        params["allThreadsRelatedToChannelId"] = all_threads_related_to_channel_id.strip()
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    moderation_status = _validate_optional_filter(moderation_status, _MODERATION_STATUSES, "moderation_status")
    order = _validate_optional_filter(order, _ORDERS, "order")
    if moderation_status:
        params["moderationStatus"] = moderation_status
    if search_terms and search_terms.strip():
        params["searchTerms"] = search_terms.strip()
    if order:
        params["order"] = order
    return params


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
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("youtube returned an invalid comments response")
    return body


async def list_comment_threads(
    user_id: str,
    *,
    video_id: str | None = None,
    channel_id: str | None = None,
    all_threads_related_to_channel_id: str | None = None,
    max_results: int = 20,
    page_token: str = "",
    moderation_status: str | None = None,
    search_terms: str | None = None,
    order: str | None = None,
    text_format: str = "plainText",
) -> dict[str, object]:
    params = _build_thread_list_params(
        video_id=video_id,
        channel_id=channel_id,
        all_threads_related_to_channel_id=all_threads_related_to_channel_id,
        max_results=max_results,
        page_token=page_token,
        moderation_status=moderation_status,
        search_terms=search_terms,
        order=order,
        text_format=text_format,
    )
    result = await _request(user_id, "GET", "/commentThreads", params=params)
    return {
        "integration": "youtube",
        "operation": "list_comment_threads",
        "comments": result.get("items") if isinstance(result.get("items"), list) else [],
        "next_page_token": result.get("nextPageToken", ""),
        "page_info": result.get("pageInfo") if isinstance(result.get("pageInfo"), dict) else {},
        "secrets_exposed": False,
    }


async def list_comment_replies(
    user_id: str,
    parent_id: str,
    *,
    max_results: int = 50,
    page_token: str = "",
    text_format: str = "plainText",
) -> dict[str, object]:
    parent_id = parent_id.strip()
    if not parent_id:
        raise ValueError("parent_id is required")
    if not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError("max_results must be between 1 and 100")
    text_format = _validate_optional_filter(text_format, _TEXT_FORMATS, "text_format") or "plainText"
    params: dict[str, object] = {
        "part": "snippet",
        "parentId": parent_id,
        "maxResults": max_results,
        "textFormat": text_format,
    }
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    result = await _request(user_id, "GET", "/comments", params=params)
    return {
        "integration": "youtube",
        "operation": "list_comment_replies",
        "parent_id": parent_id,
        "comments": result.get("items") if isinstance(result.get("items"), list) else [],
        "next_page_token": result.get("nextPageToken", ""),
        "page_info": result.get("pageInfo") if isinstance(result.get("pageInfo"), dict) else {},
        "secrets_exposed": False,
    }


async def create_top_level_comment(
    user_id: str,
    text: str,
    *,
    video_id: str | None = None,
    channel_id: str | None = None,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube comment creation")
    text = _validate_comment_text(text)
    video_id = (video_id or "").strip()
    channel_id = (channel_id or "").strip()
    if not video_id and not channel_id:
        raise ValueError("video_id or channel_id is required")
    snippet: dict[str, object] = {"topLevelComment": {"snippet": {"textOriginal": text}}}
    if video_id:
        snippet["videoId"] = video_id
    if channel_id:
        snippet["channelId"] = channel_id
    result = await _request(
        user_id,
        "POST",
        "/commentThreads",
        params={"part": "snippet"},
        json={"snippet": snippet},
    )
    return {
        "integration": "youtube",
        "operation": "create_top_level_comment",
        "comment_thread": result,
        "secrets_exposed": False,
    }


async def reply_to_comment(
    user_id: str,
    parent_id: str,
    text: str,
    *,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube comment replies")
    parent_id = parent_id.strip()
    if not parent_id:
        raise ValueError("parent_id is required")
    text = _validate_comment_text(text)
    result = await _request(
        user_id,
        "POST",
        "/comments",
        params={"part": "snippet"},
        json={"snippet": {"parentId": parent_id, "textOriginal": text}},
    )
    return {
        "integration": "youtube",
        "operation": "reply_to_comment",
        "comment": result,
        "secrets_exposed": False,
    }


async def update_comment(
    user_id: str,
    comment_id: str,
    text: str,
    *,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube comment update")
    comment_id = comment_id.strip()
    if not comment_id:
        raise ValueError("comment_id is required")
    text = _validate_comment_text(text)
    result = await _request(
        user_id,
        "PUT",
        "/comments",
        params={"part": "snippet"},
        json={"id": comment_id, "snippet": {"textOriginal": text}},
    )
    return {
        "integration": "youtube",
        "operation": "update_comment",
        "comment": result,
        "secrets_exposed": False,
    }


async def delete_comment(user_id: str, comment_id: str, *, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube comment delete")
    comment_id = comment_id.strip()
    if not comment_id:
        raise ValueError("comment_id is required")
    await _request(user_id, "DELETE", "/comments", params={"id": comment_id})
    return {
        "integration": "youtube",
        "operation": "delete_comment",
        "comment_id": comment_id,
        "deleted": True,
        "secrets_exposed": False,
    }


async def moderate_comments(
    user_id: str,
    comment_ids: list[str],
    moderation_status: str,
    *,
    ban_author: bool = False,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for youtube comment moderation")
    cleaned_ids = [item.strip() for item in comment_ids if item.strip()]
    if not cleaned_ids:
        raise ValueError("at least one comment_id is required")
    if len(cleaned_ids) > 50:
        raise ValueError("comment_ids cannot contain more than 50 comments")
    moderation_status = moderation_status.strip()
    if moderation_status not in _MODERATION_STATUSES:
        raise ValueError("moderation_status must be heldForReview, published, or rejected")
    if ban_author and moderation_status != "rejected":
        raise ValueError("ban_author is only valid when moderation_status is rejected")
    await _request(
        user_id,
        "POST",
        "/comments/setModerationStatus",
        params={
            "id": ",".join(cleaned_ids),
            "moderationStatus": moderation_status,
            "banAuthor": str(bool(ban_author)).lower(),
        },
    )
    return {
        "integration": "youtube",
        "operation": "moderate_comments",
        "comment_ids": cleaned_ids,
        "moderation_status": moderation_status,
        "ban_author": ban_author,
        "updated": True,
        "secrets_exposed": False,
    }
