from __future__ import annotations

import re
from typing import Any

import httpx

from app.capabilities.instagram import _GRAPH_URL, _access_token, _refresh_access_token, _token_row

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")

_MEDIA_FIELDS = "id,caption,media_type,media_product_type,media_url,permalink,thumbnail_url,timestamp,username"
_STORY_FIELDS = "id,media_type,media_url,permalink,thumbnail_url,timestamp"
_CONTAINER_FIELDS = "id,status,status_code"


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


async def _request_json(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, Any]:
    token = await _access_token(user_id)
    url = f"{_GRAPH_URL}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.request(method, url, params=params, headers=headers)
        if response.status_code == 401:
            refreshed = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.request(
                method,
                url,
                params=params,
                headers={"Authorization": f"Bearer {refreshed}", "Accept": "application/json"},
            )
        if response.status_code == 204:
            return {}
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram advanced response is invalid")
    return body


async def list_reels(
    user_id: str,
    limit: int = 25,
    after: str = "",
) -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    params: dict[str, object] = {"fields": _MEDIA_FIELDS, "limit": limit}
    if after.strip():
        params["after"] = after.strip()
    body = await _request_json(user_id, "GET", "me/media", params=params)
    raw_items = body.get("data", [])
    data = [
        item
        for item in raw_items
        if isinstance(item, dict)
        and (
            str(item.get("media_product_type", "")).upper() == "REELS"
            or (
                str(item.get("media_type", "")).upper() == "VIDEO"
                and "REELS" in str(item.get("media_product_type", "")).upper()
            )
        )
    ]
    return {
        "integration": "instagram",
        "media_product_type": "REELS",
        "data": data,
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }


async def list_stories(
    user_id: str,
    limit: int = 25,
    after: str = "",
) -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    params: dict[str, object] = {"fields": _STORY_FIELDS, "limit": limit}
    if after.strip():
        params["after"] = after.strip()
    body = await _request_json(user_id, "GET", "me/stories", params=params)
    return {
        "integration": "instagram",
        "media_product_type": "STORIES",
        "data": body.get("data", []),
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }


async def get_media_details(user_id: str, media_id: str) -> dict[str, object]:
    media = _validate_id(media_id, "media_id")
    body = await _request_json(user_id, "GET", media, params={"fields": _MEDIA_FIELDS})
    return {
        "integration": "instagram",
        "media_id": media,
        "media": body,
        "secrets_exposed": False,
    }


async def get_container_status(user_id: str, creation_id: str) -> dict[str, object]:
    creation = _validate_id(creation_id, "creation_id")
    body = await _request_json(user_id, "GET", creation, params={"fields": _CONTAINER_FIELDS})
    return {
        "integration": "instagram",
        "creation_id": creation,
        "container": body,
        "secrets_exposed": False,
    }
