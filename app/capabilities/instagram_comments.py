from __future__ import annotations

import re
from typing import Any

import httpx

from app.capabilities.instagram import _GRAPH_URL, _access_token, _refresh_access_token, _token_row

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MAX_MESSAGE_LENGTH = 2200


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _validate_message(message: str) -> str:
    normalized = message.strip()
    if not normalized:
        raise ValueError("message is required")
    if len(normalized) > _MAX_MESSAGE_LENGTH:
        raise ValueError("message must be at most 2200 characters")
    return normalized


async def _request_json(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    token = await _access_token(user_id)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            f"{_GRAPH_URL}/{path.lstrip('/')}",
            params=params,
            data=data,
            headers=headers,
        )
        if response.status_code == 401:
            refreshed = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.request(
                method,
                f"{_GRAPH_URL}/{path.lstrip('/')}",
                params=params,
                data=data,
                headers={"Authorization": f"Bearer {refreshed}", "Accept": "application/json"},
            )
        if response.status_code == 204:
            return {}
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram comments response is invalid")
    return body


async def list_media_comments(
    user_id: str,
    media_id: str,
    limit: int = 25,
    after: str = "",
) -> dict[str, object]:
    media = _validate_id(media_id, "media_id")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    params: dict[str, object] = {
        "fields": "id,text,username,timestamp,like_count,replies{id,text,username,timestamp}",
        "limit": limit,
    }
    if after.strip():
        params["after"] = after.strip()
    body = await _request_json(user_id, "GET", f"{media}/comments", params=params)
    return {
        "integration": "instagram",
        "media_id": media,
        "data": body.get("data", []),
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }


async def reply_to_comment(
    user_id: str,
    comment_id: str,
    message: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram comment replies require explicit approval")
    comment = _validate_id(comment_id, "comment_id")
    body = await _request_json(
        user_id,
        "POST",
        f"{comment}/replies",
        data={"message": _validate_message(message)},
    )
    return {
        "integration": "instagram",
        "comment_id": comment,
        "replied": True,
        "result": body,
        "secrets_exposed": False,
    }


async def create_media_comment(
    user_id: str,
    media_id: str,
    message: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram comment creation requires explicit approval")
    media = _validate_id(media_id, "media_id")
    body = await _request_json(
        user_id,
        "POST",
        f"{media}/comments",
        data={"message": _validate_message(message)},
    )
    return {
        "integration": "instagram",
        "media_id": media,
        "created": True,
        "result": body,
        "secrets_exposed": False,
    }


async def delete_comment(user_id: str, comment_id: str, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram comment deletion requires explicit approval")
    comment = _validate_id(comment_id, "comment_id")
    await _request_json(user_id, "DELETE", comment)
    return {
        "integration": "instagram",
        "comment_id": comment,
        "deleted": True,
        "secrets_exposed": False,
    }


async def set_comment_hidden(
    user_id: str,
    comment_id: str,
    hidden: bool,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram comment moderation requires explicit approval")
    comment = _validate_id(comment_id, "comment_id")
    await _request_json(
        user_id,
        "POST",
        comment,
        data={"hide": "true" if hidden else "false"},
    )
    return {
        "integration": "instagram",
        "comment_id": comment,
        "hidden": hidden,
        "moderated": True,
        "secrets_exposed": False,
    }
