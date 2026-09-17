from __future__ import annotations

import re
from typing import Any

import httpx

from app.capabilities.instagram import _GRAPH_URL, _access_token, _refresh_access_token, _token_row

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MAX_CAROUSEL_ITEMS = 10


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _validate_image_urls(image_urls: list[str]) -> list[str]:
    if not 2 <= len(image_urls) <= _MAX_CAROUSEL_ITEMS:
        raise ValueError("image_urls must contain between 2 and 10 items")
    normalized: list[str] = []
    for value in image_urls:
        url = value.strip()
        if not url:
            raise ValueError("image_urls cannot contain empty values")
        normalized.append(url)
    return normalized


def _validate_caption(caption: str) -> str:
    normalized = caption.strip()
    if len(normalized) > 2200:
        raise ValueError("caption must be at most 2200 characters")
    return normalized


async def _request_json(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
) -> dict[str, Any]:
    token = await _access_token(user_id)
    url = f"{_GRAPH_URL}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, url, params=params, data=data, headers=headers)
        if response.status_code == 401:
            refreshed = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.request(
                method,
                url,
                params=params,
                data=data,
                headers={"Authorization": f"Bearer {refreshed}", "Accept": "application/json"},
            )
        if response.status_code == 204:
            return {}
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram publishing response is invalid")
    return body


async def create_carousel_container(
    user_id: str,
    image_urls: list[str],
    caption: str = "",
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram carousel creation requires explicit approval")
    urls = _validate_image_urls(image_urls)
    normalized_caption = _validate_caption(caption)
    token = await _access_token(user_id)

    child_ids: list[str] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for image_url in urls:
            response = await client.post(
                f"{_GRAPH_URL}/me/media",
                data={
                    "image_url": image_url,
                    "is_carousel_item": "true",
                    "access_token": token,
                },
                headers={"Accept": "application/json"},
            )
            if response.status_code == 401:
                token = await _refresh_access_token(user_id, _token_row(user_id))
                response = await client.post(
                    f"{_GRAPH_URL}/me/media",
                    data={
                        "image_url": image_url,
                        "is_carousel_item": "true",
                        "access_token": token,
                    },
                    headers={"Accept": "application/json"},
                )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("id"), str):
                raise RuntimeError("instagram carousel child container response is invalid")
            child_ids.append(body["id"])

        parent_data: dict[str, object] = {
            "media_type": "CAROUSEL",
            "children": ",".join(child_ids),
            "access_token": token,
        }
        if normalized_caption:
            parent_data["caption"] = normalized_caption
        response = await client.post(
            f"{_GRAPH_URL}/me/media",
            data=parent_data,
            headers={"Accept": "application/json"},
        )
        if response.status_code == 401:
            token = await _refresh_access_token(user_id, _token_row(user_id))
            parent_data["access_token"] = token
            response = await client.post(
                f"{_GRAPH_URL}/me/media",
                data=parent_data,
                headers={"Accept": "application/json"},
            )
        response.raise_for_status()
        parent_body = response.json()

    if not isinstance(parent_body, dict) or not isinstance(parent_body.get("id"), str):
        raise RuntimeError("instagram carousel container response is invalid")
    return {
        "integration": "instagram",
        "container": parent_body,
        "child_containers": child_ids,
        "item_count": len(child_ids),
        "secrets_exposed": False,
    }


async def publish_ready_container(
    user_id: str,
    creation_id: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram publishing requires explicit approval")
    creation = _validate_id(creation_id, "creation_id")
    status = await _request_json(
        user_id,
        "GET",
        creation,
        params={"fields": "id,status,status_code"},
    )
    container_status = str(status.get("status", "")).upper()
    if container_status != "FINISHED":
        raise ValueError(f"instagram container is not ready for publishing: {container_status or 'UNKNOWN'}")

    body = await _request_json(
        user_id,
        "POST",
        "me/media_publish",
        data={"creation_id": creation, "access_token": await _access_token(user_id)},
    )
    return {
        "integration": "instagram",
        "creation_id": creation,
        "published": True,
        "container_status": container_status,
        "result": body,
        "secrets_exposed": False,
    }
