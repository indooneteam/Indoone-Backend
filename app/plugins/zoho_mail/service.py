"""Zoho Mail API wrapper using encrypted user-scoped OAuth tokens."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
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
    if not isinstance(provider_status, dict):
        raise RuntimeError("Zoho Mail API response omitted its status object")
    code = provider_status.get("code")
    if str(code) not in {"200", "201", "202"}:
        description = str(provider_status.get("description") or "request failed")
        raise RuntimeError(f"Zoho Mail API failed: {description[:200]}")
    return result


async def list_accounts(user_id: str) -> dict[str, Any]:
    return await _request(user_id, "GET", "accounts")


async def list_folders(user_id: str, account_id: str) -> dict[str, Any]:
    account = _identifier(account_id, "account_id")
    return await _request(user_id, "GET", f"accounts/{account}/folders")


_ZOHO_SEARCH_FIELDS = {
    "entire": "entire",
    "content": "content",
    "sender": "sender",
    "to": "to",
    "cc": "cc",
    "subject": "subject",
    "filename": "fileName",
    "filecontent": "fileContent",
    "has": "has",
    "in": "in",
    "label": "label",
    "fromdate": "fromDate",
    "todate": "toDate",
    "inclspamtrash": "inclspamtrash",
    "groupresult": "groupResult",
}
_ZOHO_SEARCH_ALIASES = {
    "from": "sender",
    "folder": "in",
    "tag": "label",
}
_ZOHO_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _normalize_search_date(value: str) -> str:
    raw = value.strip().strip('"')
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[Tt].*)?", raw):
            parsed = date.fromisoformat(raw[:10])
        else:
            parsed = datetime.strptime(raw, "%d-%b-%Y").date()
    except ValueError as exc:
        raise ValueError(
            "Zoho Mail date filters require YYYY-MM-DD or DD-MMM-YYYY"
        ) from exc
    return f"{parsed.day:02d}-{_ZOHO_MONTHS[parsed.month - 1]}-{parsed.year:04d}"


def _quoted_search_value(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        return normalized
    return '"' + normalized.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _normalize_search_clause(clause: str) -> str:
    value = clause.strip()
    if not value:
        raise ValueError("search_key contains an empty Zoho Mail search condition")
    if value.casefold().startswith("or:"):
        return "or:" + _normalize_search_clause(value[3:])

    match = re.match(r"^([a-zA-Z][a-zA-Z0-9]*):(.*)$", value)
    if not match:
        return "entire:" + _quoted_search_value(value)

    field = match.group(1).casefold()
    term = match.group(2).strip()
    if not term:
        raise ValueError(f"Zoho Mail search filter '{field}:' requires a value")
    if field == "bcc":
        raise ValueError(
            "Zoho Mail does not support bcc: search filters; use sender:, to:, cc:, subject:, or a general search"
        )
    if field in {"before", "after", "on", "date"}:
        formatted_date = _normalize_search_date(term)
        if field == "before":
            return f"toDate:{formatted_date}"
        if field == "after":
            return f"fromDate:{formatted_date}"
        return f"fromDate:{formatted_date}::toDate:{formatted_date}"

    canonical = _ZOHO_SEARCH_ALIASES.get(field) or _ZOHO_SEARCH_FIELDS.get(field)
    if canonical is None:
        return "entire:" + _quoted_search_value(value)

    if canonical in {"entire", "content", "subject", "sender", "to", "cc", "fileName", "fileContent", "in", "label"}:
        term = _quoted_search_value(term) if re.search(r"\s", term) else term
    return f"{canonical}:{term}"


def _normalize_search_key(search_key: str) -> str:
    raw = search_key.strip()
    if not raw:
        raise ValueError("search_key is required")
    if len(raw) > 512:
        raise ValueError("search_key exceeds maximum length of 512")
    normalized = "::".join(_normalize_search_clause(part) for part in raw.split("::"))
    if len(normalized) > 512:
        raise ValueError("normalized search_key exceeds maximum length of 512")
    return normalized


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
        params["searchKey"] = _normalize_search_key(search_key)
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
    created_data = created.get("data") if isinstance(created, dict) else None
    if isinstance(created_data, list):
        created_data = created_data[0] if created_data and isinstance(created_data[0], dict) else None
    new_draft_id = (
        str(
            created_data.get("messageId")
            or created_data.get("message_id")
            or created_data.get("id")
            or ""
        ).strip()
        if isinstance(created_data, dict)
        else ""
    )
    if not new_draft_id:
        # Never remove the original unless Zoho confirms the replacement draft
        # with an ID that can be inspected or cleaned up if necessary.
        raise RuntimeError(
            "Zoho Mail did not return a verifiable new draft message ID; the original draft was kept"
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
        "new_draft_id": new_draft_id,
        "replaced_message_id": draft_id,
        "secrets_exposed": False,
    }
