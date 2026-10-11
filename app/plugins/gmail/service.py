from __future__ import annotations

import base64
from email.message import EmailMessage

import httpx
from app.capabilities.integrations import get_valid_access_token

_GMAIL_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


async def _token(user_id: str) -> str:
    return await get_valid_access_token(user_id, "gmail")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _raw_message(to: str, subject: str, body: str) -> str:
    message = EmailMessage()
    if to.strip():
        message["To"] = to.strip()
    if subject.strip():
        message["Subject"] = subject.strip()
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")


async def list_gmail_drafts(user_id: str, max_results: int = 20) -> dict[str, object]:
    if not user_id.strip():
        raise ValueError("user_id is required")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    token = await _token(user_id)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GMAIL_BASE_URL}/drafts",
            headers=_headers(token),
            params={"maxResults": max_results},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("gmail provider returned invalid draft list")
    drafts = body.get("drafts")
    return {
        "integration": "gmail",
        "user_id": user_id.strip(),
        "drafts": drafts if isinstance(drafts, list) else [],
        "result_size_estimate": body.get("resultSizeEstimate", 0),
        "secrets_exposed": False,
    }


async def create_gmail_draft(
    user_id: str,
    to: str = "",
    subject: str = "",
    body: str = "",
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    if len(to) > 512 or len(subject) > 2000 or len(body) > 20000:
        raise ValueError("draft fields exceed their maximum allowed length")
    if not body.strip() and not subject.strip() and not to.strip():
        raise ValueError("draft content is required")
    token = await _token(user)
    payload = {"message": {"raw": _raw_message(to, subject, body)}}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GMAIL_BASE_URL}/drafts",
            headers=_headers(token),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("gmail provider returned invalid draft response")
    return {
        "integration": "gmail",
        "user_id": user,
        "operation": "create_draft",
        "draft": {
            "id": result.get("id"),
            "message_id": (result.get("message") or {}).get("id")
            if isinstance(result.get("message"), dict)
            else None,
        },
        "secrets_exposed": False,
    }


async def update_gmail_draft(
    user_id: str,
    draft_id: str,
    to: str = "",
    subject: str = "",
    body: str = "",
) -> dict[str, object]:
    user = user_id.strip()
    normalized_draft_id = draft_id.strip()
    if not user or not normalized_draft_id or len(normalized_draft_id) > 256:
        raise ValueError("user_id and a valid draft_id are required")
    if len(to) > 512 or len(subject) > 2000 or len(body) > 20000:
        raise ValueError("draft fields exceed their maximum allowed length")
    token = await _token(user)
    payload = {"message": {"raw": _raw_message(to, subject, body)}}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.put(
            f"{_GMAIL_BASE_URL}/drafts/{normalized_draft_id}",
            headers=_headers(token),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("gmail provider returned invalid updated draft response")
    return {
        "integration": "gmail",
        "user_id": user,
        "operation": "update_draft",
        "draft_id": result.get("id", normalized_draft_id),
        "secrets_exposed": False,
    }


async def delete_gmail_draft(
    user_id: str,
    draft_id: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for Gmail draft deletion")
    user = user_id.strip()
    normalized_draft_id = draft_id.strip()
    if not user or not normalized_draft_id or len(normalized_draft_id) > 256:
        raise ValueError("user_id and a valid draft_id are required")
    token = await _token(user)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.delete(
            f"{_GMAIL_BASE_URL}/drafts/{normalized_draft_id}",
            headers=_headers(token),
        )
        response.raise_for_status()
    return {
        "integration": "gmail",
        "user_id": user,
        "operation": "delete_draft",
        "draft_id": normalized_draft_id,
        "secrets_exposed": False,
    }


async def delete_gmail_message(
    user_id: str,
    message_id: str,
    approved: bool = False,
) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for Gmail message deletion")
    user = user_id.strip()
    normalized_message_id = message_id.strip()
    if not user or not normalized_message_id or len(normalized_message_id) > 256:
        raise ValueError("user_id and a valid message_id are required")
    token = await _token(user)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GMAIL_BASE_URL}/messages/{normalized_message_id}/trash",
            headers=_headers(token),
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("gmail provider returned invalid trash response")
    return {
        "integration": "gmail",
        "user_id": user,
        "operation": "trash_message",
        "message_id": normalized_message_id,
        "secrets_exposed": False,
    }



async def list_gmail_messages(user_id: str, query: str = "", page_token: str = "", max_results: int = 20) -> dict[str, object]:
    if not user_id.strip():
        raise ValueError("user_id is required")
    if len(query) > 500:
        raise ValueError("query exceeds maximum length of 500")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    token = await _token(user_id)
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
    token = await _token(user_id)
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
    token = await _token(user)
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
