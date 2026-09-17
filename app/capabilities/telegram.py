from __future__ import annotations

import os
from typing import Any

import httpx

_API_BASE = "https://api.telegram.org/bot"
_MAX_MESSAGE_LENGTH = 4096
_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024


def _token() -> str:
    token = os.getenv("INDOONE_TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("telegram bot token is not configured")
    return token


def _secret_token() -> str:
    return os.getenv("INDOONE_TELEGRAM_WEBHOOK_SECRET", "").strip()


def _url(method: str) -> str:
    return f"{_API_BASE}{_token()}/{method}"


async def _call(method: str, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(_url(method), json=payload or {})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict) or body.get("ok") is not True:
        detail = body.get("description") if isinstance(body, dict) else "invalid telegram response"
        raise RuntimeError(str(detail or "telegram api request failed"))
    return body.get("result")


async def probe_telegram() -> dict[str, object]:
    result = await _call("getMe")
    if not isinstance(result, dict):
        raise RuntimeError("telegram returned an invalid bot profile")
    return {
        "integration": "telegram",
        "configured": True,
        "provider_ok": True,
        "bot": {
            "id": result.get("id"),
            "username": result.get("username"),
            "first_name": result.get("first_name"),
            "can_join_groups": result.get("can_join_groups"),
            "can_read_all_group_messages": result.get("can_read_all_group_messages"),
        },
        "secrets_exposed": False,
    }


async def get_updates(offset: int | None = None, limit: int = 100, timeout: int = 0) -> dict[str, object]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if timeout < 0 or timeout > 50:
        raise ValueError("timeout must be between 0 and 50")
    payload: dict[str, Any] = {"limit": limit, "timeout": timeout}
    if offset is not None:
        payload["offset"] = offset
    result = await _call("getUpdates", payload, timeout=max(20.0, timeout + 10.0))
    return {
        "integration": "telegram",
        "updates": result if isinstance(result, list) else [],
        "secrets_exposed": False,
    }


async def send_message(chat_id: str, text: str, approved: bool = False, parse_mode: str = "") -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for telegram send operations")
    chat_id = chat_id.strip()
    text = text.strip()
    if not chat_id:
        raise ValueError("chat_id is required")
    if not text:
        raise ValueError("text is required")
    if len(text) > _MAX_MESSAGE_LENGTH:
        raise ValueError("text exceeds Telegram's 4096-character message limit")
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode.strip():
        if parse_mode.strip() not in {"Markdown", "MarkdownV2", "HTML"}:
            raise ValueError("parse_mode must be Markdown, MarkdownV2, or HTML")
        payload["parse_mode"] = parse_mode.strip()
    result = await _call("sendMessage", payload)
    return {
        "integration": "telegram",
        "operation": "send_message",
        "message": result,
        "secrets_exposed": False,
    }


async def get_webhook_info() -> dict[str, object]:
    result = await _call("getWebhookInfo")
    return {
        "integration": "telegram",
        "webhook": result if isinstance(result, dict) else {},
        "secrets_exposed": False,
    }


async def set_webhook(url: str, approved: bool = False, drop_pending_updates: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for telegram webhook changes")
    url = url.strip()
    if not url.startswith("https://"):
        raise ValueError("telegram webhook url must use HTTPS")
    payload: dict[str, Any] = {
        "url": url,
        "drop_pending_updates": drop_pending_updates,
    }
    secret = _secret_token()
    if secret:
        payload["secret_token"] = secret
    result = await _call("setWebhook", payload)
    return {
        "integration": "telegram",
        "operation": "set_webhook",
        "result": result,
        "secret_configured": bool(secret),
        "secrets_exposed": False,
    }


def validate_webhook_secret(received: str | None) -> bool:
    configured = _secret_token()
    return bool(configured and received and received == configured)


async def delete_webhook(approved: bool = False, drop_pending_updates: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for telegram webhook changes")
    result = await _call("deleteWebhook", {"drop_pending_updates": drop_pending_updates})
    return {
        "integration": "telegram",
        "operation": "delete_webhook",
        "result": result,
        "secrets_exposed": False,
    }
