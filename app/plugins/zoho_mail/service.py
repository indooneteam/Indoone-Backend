"""Zoho Mail API wrapper using encrypted user-scoped OAuth tokens."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from app.capabilities.integrations import get_valid_access_token


def _mail_api_base_url() -> str:
    value = os.getenv("INDOONE_ZOHO_MAIL_API_BASE_URL", "https://mail.zoho.in/api").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("INDOONE_ZOHO_MAIL_API_BASE_URL must be a valid HTTPS URL.")
    return value


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 256 or any(char in normalized for char in ("/", "\\", "\\x00")):
        raise ValueError(f"{field_name} is invalid")
    return quote(normalized, safe="")


async def _request(
    user_id: str,
    method: str,
    path: str,
    *,
    params: dict[str, str | int | bool] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")
    token = await get_valid_access_token(normalized_user_id, "zoho_mail")
    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "Accept": "application/json",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.request(
            method,
            f"{_mail_api_base_url()}/{path.lstrip('/')}",
            headers=headers,
            params=params,
            json=payload,
        )
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("Zoho Mail API returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Zoho Mail API returned an unexpected response")
    provider_status = result.get("status")
    if isinstance(provider_status, dict):
        code = provider_status.get("code")
        if code is not None and str(code) not in {"200", "201", "202"}:
            description = str(provider_status.get("description") or "request failed")
            raise RuntimeError(f"Zoho Mail API failed: {description[:200]}")
    return result


async def list_accounts(user_id: str) -> dict[str, Any]:
    return await _request(user_id, "GET", "accounts")


async def list_folders(user_id: str, account_id: str) -> dict[str, Any]:
    account = _identifier(account_id, "account_id")
    return await _request(user_id, "GET", f"accounts/{account}/folders")


async def list_messages(
    user_id: str,
    account_id: str,
    *,
    folder_id: str = "",
    search_key: str = "",
    start: int = 1,
    limit: int = 20,
) -> dict[str, Any]:
    account = _identifier(account_id, "account_id")
    if start < 1:
        raise ValueError("start must be at least 1")
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    params: dict[str, str | int | bool] = {"start": start, "limit": limit, "includeto": True}
    if search_key.strip():
        if len(search_key) > 512:
            raise ValueError("search_key exceeds maximum length of 512")
        params["searchKey"] = search_key.strip()
        return await _request(user_id, "GET", f"accounts/{account}/messages/search", params=params)
    if folder_id.strip():
        params["folderId"] = _identifier(folder_id, "folder_id")
    return await _request(user_id, "GET", f"accounts/{account}/messages/view", params=params)


async def get_message_content(
    user_id: str,
    account_id: str,
    folder_id: str,
    message_id: str,
) -> dict[str, Any]:
    account = _identifier(account_id, "account_id")
    folder = _identifier(folder_id, "folder_id")
    message = _identifier(message_id, "message_id")
    return await _request(
        user_id,
        "GET",
        f"accounts/{account}/folders/{folder}/messages/{message}/content",
    )


async def send_message(
    user_id: str,
    account_id: str,
    from_address: str,
    to_address: str,
    subject: str,
    content: str,
    approved: bool = False,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("explicit approval is required for Zoho Mail send operations")
    account = _identifier(account_id, "account_id")
    sender = from_address.strip()
    recipient = to_address.strip()
    normalized_subject = subject.strip()
    if not sender or len(sender) > 512 or "@" not in sender:
        raise ValueError("from_address must be a valid email address")
    if not recipient or len(recipient) > 512 or "@" not in recipient:
        raise ValueError("to_address must be a valid email address")
    if not normalized_subject or len(normalized_subject) > 2_000:
        raise ValueError("subject is required and must not exceed 2000 characters")
    if not content or len(content) > 20_000:
        raise ValueError("content is required and must not exceed 20000 characters")
    payload = {
        "fromAddress": sender,
        "toAddress": recipient,
        "subject": normalized_subject,
        "content": content,
        "mailFormat": "plaintext",
    }
    return await _request(user_id, "POST", f"accounts/{account}/messages", payload=payload)


async def reply_to_message(
    user_id: str,
    account_id: str,
    message_id: str,
    from_address: str,
    to_address: str,
    subject: str,
    content: str,
    approved: bool = False,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("explicit approval is required for Zoho Mail reply operations")
    account = _identifier(account_id, "account_id")
    message = _identifier(message_id, "message_id")
    sender = from_address.strip()
    recipient = to_address.strip()
    normalized_subject = subject.strip()
    if not sender or len(sender) > 512 or "@" not in sender:
        raise ValueError("from_address must be a valid email address")
    if not recipient or len(recipient) > 512 or "@" not in recipient:
        raise ValueError("to_address must be a valid email address")
    if not normalized_subject or len(normalized_subject) > 2_000:
        raise ValueError("subject is required and must not exceed 2000 characters")
    if not content or len(content) > 20_000:
        raise ValueError("content is required and must not exceed 20000 characters")
    payload = {
        "fromAddress": sender,
        "toAddress": recipient,
        "subject": normalized_subject,
        "content": content,
        "action": "reply",
        "mailFormat": "plaintext",
    }
    return await _request(
        user_id,
        "POST",
        f"accounts/{account}/messages/{message}",
        payload=payload,
    )



async def delete_message(
    user_id: str,
    account_id: str,
    folder_id: str,
    message_id: str,
    *,
    approved: bool = False,
    permanently: bool = False,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("explicit approval is required for Zoho Mail delete operations")
    account = _identifier(account_id, "account_id")
    folder = _identifier(folder_id, "folder_id")
    message = _identifier(message_id, "message_id")
    return await _request(
        user_id,
        "DELETE",
        f"accounts/{account}/folders/{folder}/messages/{message}",
        params={"expunge": permanently},
    )


async def save_draft(
    user_id: str,
    account_id: str,
    from_address: str,
    to_address: str,
    subject: str,
    content: str,
) -> dict[str, Any]:
    account = _identifier(account_id, "account_id")
    sender = from_address.strip()
    recipient = to_address.strip()
    normalized_subject = subject.strip()
    if not sender or len(sender) > 512 or "@" not in sender:
        raise ValueError("from_address must be a valid email address")
    if recipient and (len(recipient) > 512 or "@" not in recipient):
        raise ValueError("to_address must be a valid email address when provided")
    if len(normalized_subject) > 2_000:
        raise ValueError("subject must not exceed 2000 characters")
    if len(content) > 20_000:
        raise ValueError("content must not exceed 20000 characters")
    payload = {
        "mode": "draft",
        "fromAddress": sender,
        "toAddress": recipient,
        "subject": normalized_subject,
        "content": content,
        "mailFormat": "plaintext",
    }
    return await _request(user_id, "POST", f"accounts/{account}/messages", payload=payload)


async def replace_draft(
    user_id: str,
    account_id: str,
    folder_id: str,
    draft_id: str,
    from_address: str,
    to_address: str,
    subject: str,
    content: str,
    *,
    approved: bool = False,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("explicit approval is required to replace a Zoho Mail draft")
    # Zoho Mail's documented message API exposes draft creation and message deletion.
    # Replacing a draft creates the new copy first, then removes the old draft; this
    # avoids losing the original if draft creation fails.
    created = await save_draft(
        user_id, account_id, from_address, to_address, subject, content
    )
    await delete_message(
        user_id,
        account_id,
        folder_id,
        draft_id,
        approved=True,
        permanently=False,
    )
    return {
        "integration": "zoho_mail",
        "operation": "replace_draft",
        "created": created,
        "replaced_message_id": draft_id,
        "secrets_exposed": False,
    }
