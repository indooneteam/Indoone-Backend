from __future__ import annotations

import os
from typing import Any

import httpx

_API_BASE = "https://graph.facebook.com"
_API_VERSION = os.getenv("INDOONE_WHATSAPP_GRAPH_VERSION", "v23.0")
_MAX_TEXT_LENGTH = 4096


def _token() -> str:
    token = os.getenv("INDOONE_WHATSAPP_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("whatsapp access token is not configured")
    return token


def _phone_number_id() -> str:
    value = os.getenv("INDOONE_WHATSAPP_PHONE_NUMBER_ID", "").strip()
    if not value:
        raise RuntimeError("whatsapp phone number id is not configured")
    return value


def _webhook_verify_token() -> str:
    return os.getenv("INDOONE_WHATSAPP_WEBHOOK_VERIFY_TOKEN", "").strip()


def _url(path: str) -> str:
    return f"{_API_BASE}/{_API_VERSION}/{path.lstrip('/')}"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}


async def probe_whatsapp() -> dict[str, object]:
    phone_id = _phone_number_id()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(_url(phone_id), headers=_headers(), params={"fields": "id,display_phone_number,verified_name,quality_rating"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("whatsapp returned an invalid phone number response")
    return {"integration": "whatsapp_business", "configured": True, "provider_ok": True, "phone_number": body, "secrets_exposed": False}


async def send_text_message(to: str, text: str, approved: bool = False, preview_url: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for whatsapp send operations")
    recipient = to.strip()
    message = text.strip()
    if not recipient:
        raise ValueError("to is required")
    if not message:
        raise ValueError("text is required")
    if len(message) > _MAX_TEXT_LENGTH:
        raise ValueError("text exceeds WhatsApp Cloud API message limit")
    payload: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": bool(preview_url), "body": message},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(_url(f"{_phone_number_id()}/messages"), headers=_headers(), json=payload)
        response.raise_for_status()
        body = response.json()
    return {"integration": "whatsapp_business", "operation": "send_text", "result": body, "secrets_exposed": False}


def verify_webhook(mode: str | None, verify_token: str | None, challenge: str | None) -> str:
    configured = _webhook_verify_token()
    if mode != "subscribe" or not configured or verify_token != configured or not challenge:
        raise PermissionError("invalid whatsapp webhook verification request")
    return challenge


def validate_signature(app_secret: str, signature_header: str | None, raw_body: bytes) -> bool:
    if not app_secret.strip() or not signature_header or not signature_header.startswith("sha256="):
        return False
    import hashlib
    import hmac
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    supplied = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, supplied)


def parse_webhook(payload: dict[str, Any]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for entry in payload.get("entry", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes", []):
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            for message in value.get("messages", []):
                if not isinstance(message, dict):
                    continue
                messages.append({"id": message.get("id"), "from": message.get("from"), "timestamp": message.get("timestamp"), "type": message.get("type"), "text": (message.get("text") or {}).get("body") if isinstance(message.get("text"), dict) else None, "raw": message})
    return messages
