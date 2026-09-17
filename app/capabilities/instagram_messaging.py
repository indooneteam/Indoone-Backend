from __future__ import annotations

import re
from typing import Any

import httpx

from app.capabilities.instagram import _GRAPH_URL, _access_token, _refresh_access_token, _token_row

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MAX_TEXT_LENGTH = 1000


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _validate_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("text is required")
    if len(normalized) > _MAX_TEXT_LENGTH:
        raise ValueError("text must be at most 1000 characters")
    return normalized


async def _request_json(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    json: dict[str, object] | None = None,
) -> dict[str, Any]:
    token = await _access_token(user_id)
    url = f"{_GRAPH_URL}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.request(method, url, params=params, json=json, headers=headers)
        if response.status_code == 401:
            refreshed = await _refresh_access_token(user_id, _token_row(user_id))
            response = await client.request(
                method,
                url,
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {refreshed}", "Accept": "application/json"},
            )
        if response.status_code == 204:
            return {}
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("instagram messaging response is invalid")
    return body


async def list_conversations(
    user_id: str,
    limit: int = 25,
    after: str = "",
    target_user_id: str = "",
) -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    params: dict[str, object] = {"platform": "instagram", "limit": limit}
    if after.strip():
        params["after"] = after.strip()
    if target_user_id.strip():
        params["user_id"] = _validate_id(target_user_id, "target_user_id")
    body = await _request_json(user_id, "GET", "me/conversations", params=params)
    return {
        "integration": "instagram",
        "data": body.get("data", []),
        "paging": body.get("paging"),
        "secrets_exposed": False,
    }


async def list_conversation_messages(
    user_id: str,
    conversation_id: str,
) -> dict[str, object]:
    conversation = _validate_id(conversation_id, "conversation_id")
    body = await _request_json(
        user_id,
        "GET",
        conversation,
        params={"fields": "messages"},
    )
    return {
        "integration": "instagram",
        "conversation_id": conversation,
        "messages": body.get("messages", {}),
        "secrets_exposed": False,
    }


async def get_message(
    user_id: str,
    message_id: str,
) -> dict[str, object]:
    message = _validate_id(message_id, "message_id")
    body = await _request_json(
        user_id,
        "GET",
        message,
        params={"fields": "id,created_time,from,to,message"},
    )
    return {
        "integration": "instagram",
        "message": body,
        "secrets_exposed": False,
    }


async def send_text_message(
    user_id: str,
    recipient_id: str,
    text: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram messages require explicit approval")
    recipient = _validate_id(recipient_id, "recipient_id")
    body = await _request_json(
        user_id,
        "POST",
        "me/messages",
        json={"recipient": {"id": recipient}, "message": {"text": _validate_text(text)}},
    )
    return {
        "integration": "instagram",
        "recipient_id": recipient,
        "sent": True,
        "result": body,
        "secrets_exposed": False,
    }


async def send_media_share(
    user_id: str,
    recipient_id: str,
    media_id: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("instagram messages require explicit approval")
    recipient = _validate_id(recipient_id, "recipient_id")
    media = _validate_id(media_id, "media_id")
    body = await _request_json(
        user_id,
        "POST",
        "me/messages",
        json={
            "recipient": {"id": recipient},
            "message": {"attachment": {"type": "MEDIA_SHARE", "payload": {"id": media}}},
        },
    )
    return {
        "integration": "instagram",
        "recipient_id": recipient,
        "media_id": media,
        "sent": True,
        "result": body,
        "secrets_exposed": False,
    }
