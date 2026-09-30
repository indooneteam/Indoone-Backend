from __future__ import annotations

import base64
from email.message import EmailMessage
import os

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import get_integration_token

_GMAIL_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


def _token(user_id: str) -> str:
    row = get_integration_token(user_id.strip(), "gmail")
    if row is None:
        raise ValueError("integration is not connected for user")
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii")).decrypt(bytes(row["access_token"])).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored oauth token cannot be decrypted") from exc


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def list_gmail_messages(user_id: str, query: str = "", page_token: str = "", max_results: int = 20) -> dict[str, object]:
    if not user_id.strip():
        raise ValueError("user_id is required")
    if len(query) > 500:
        raise ValueError("query exceeds maximum length of 500")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    token = _token(user_id)
    params: dict[str, object] = {"maxResults": max_results}
    if query.strip():
        params["q"] = query.strip()
    if page_token.strip():
        if len(page_token) > 2048:
            raise ValueError("page_token exceeds maximum length of 2048")
        params["pageToken"] = page_token.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GMAIL_BASE_URL}/messages", headers=_headers(token), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("gmail provider returned invalid message list")
    messages = body.get("messages")
    return {
        "integration": "gmail",
        "user_id": user_id.strip(),
        "messages": messages if isinstance(messages, list) else [],
        "next_page_token": body.get("nextPageToken"),
        "result_size_estimate": body.get("resultSizeEstimate", 0),
        "secrets_exposed": False,
    }


async def get_gmail_message(user_id: str, message_id: str) -> dict[str, object]:
    if not user_id.strip():
        raise ValueError("user_id is required")
    message_id = message_id.strip()
    if not message_id or len(message_id) > 256:
        raise ValueError("message_id is required and must be at most 256 characters")
    token = _token(user_id)
    params = {"format": "full"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_GMAIL_BASE_URL}/messages/{message_id}", headers=_headers(token), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("gmail provider returned invalid message")
    return {"integration": "gmail", "user_id": user_id.strip(), "message": body, "secrets_exposed": False}


async def _send_raw_message(
    user_id: str,
    *,
    to: str,
    subject: str,
    body: str,
    approved: bool,
    thread_id: str = "",
    in_reply_to: str = "",
    references: str = "",
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    if not approved:
        raise PermissionError("explicit approval is required for gmail send operations")

    recipient = to.strip()
    subject = subject.strip()
    content = body.strip()
    if not recipient:
        raise ValueError("to is required")
    if not subject:
        raise ValueError("subject is required")
    if len(recipient) > 512:
        raise ValueError("to exceeds maximum length of 512")
    if len(subject) > 2000:
        raise ValueError("subject exceeds maximum length of 2000")
    if not content:
        raise ValueError("body is required")
    if len(content) > 20000:
        raise ValueError("body exceeds maximum length of 20000")

    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
    if references:
        message["References"] = references
    message.set_content(content)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
    token = _token(user)
    payload: dict[str, object] = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GMAIL_BASE_URL}/messages/send",
            headers=_headers(token),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("gmail provider returned invalid send response")
    return {
        "integration": "gmail",
        "user_id": user,
        "operation": "reply" if in_reply_to else "send",
        "result": {"id": result.get("id"), "thread_id": result.get("threadId")},
        "secrets_exposed": False,
    }


async def send_gmail_message(
    user_id: str,
    to: str,
    subject: str,
    body: str,
    approved: bool = False,
) -> dict[str, object]:
    return await _send_raw_message(
        user_id,
        to=to,
        subject=subject,
        body=body,
        approved=approved,
    )


async def reply_gmail_message(
    user_id: str,
    message_id: str,
    body: str,
    approved: bool = False,
) -> dict[str, object]:
    original = await get_gmail_message(user_id, message_id)
    message = original.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("gmail provider returned an invalid original message")

    headers: dict[str, str] = {}
    payload = message.get("payload")
    raw_headers = payload.get("headers", []) if isinstance(payload, dict) else []
    if isinstance(raw_headers, list):
        for item in raw_headers:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip().casefold()
            value = str(item.get("value", "")).strip()
            if name and value:
                headers[name] = value

    sender = headers.get("from", "").strip()
    original_subject = headers.get("subject", "").strip()
    message_id_header = headers.get("message-id", "").strip()
    existing_references = headers.get("references", "").strip()
    thread_id = str(message.get("threadId", "")).strip()

    if not sender:
        raise ValueError("original email sender is missing")

    subject = original_subject
    if subject and not subject.casefold().startswith("re:"):
        subject = f"Re: {subject}"
    if not subject:
        subject = "Re:"

    references = " ".join(
        part for part in (existing_references, message_id_header) if part
    ).strip()

    return await _send_raw_message(
        user_id,
        to=sender,
        subject=subject,
        body=body,
        approved=approved,
        thread_id=thread_id,
        in_reply_to=message_id_header,
        references=references,
    )
