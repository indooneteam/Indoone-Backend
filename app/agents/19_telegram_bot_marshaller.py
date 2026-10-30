"""Agent 19 - the Telegram door: same discipline as WhatsApp, different shape.

Telegram sends one ``update`` object with a different body for each kind of thing that
happened: a message, a slash command, a button press on an inline keyboard, a voice note, or
a channel post nobody asked us about. Every consumer downstream would otherwise have to know
all of it.

So this verifies, flattens and de-duplicates, exactly as the WhatsApp door does:

* **The secret token is the door.** Telegram sends ``X-Telegram-Bot-Api-Secret-Token`` on
  every webhook, and it is compared in constant time - a comparison that returns early leaks
  how much of the token was right, one character at a time.
* **``update_id`` is the retry guard.** Telegram re-sends an update when the reply is slow.
  A bounded window of recent ids drops the retry, so one message never becomes two answers.
* **A command is a command.** ``/start arg1`` becomes ``command="start"``,
  ``command_args="arg1"`` - not a sentence with a slash in it.
* **A button press is not a message.** ``callback_query`` carries the action chip the user
  pressed and the id Telegram needs to close the spinner, and both are kept.

It never downloads a file: a photo or a voice note arrives as a pointer, and the highest
resolution one where Telegram sends several sizes.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

DEDUP_MAXSIZE = 1024
DEDUP_TTL_SECONDS = 300.0
MAX_TEXT_CHARS = 4_000

_COMMAND = re.compile(r"^/(?P<command>[A-Za-z0-9_]{1,64})(?:@[A-Za-z0-9_]+)?(?:\s+(?P<args>.*))?$", re.DOTALL)

_MESSAGES = {
    "BAD_SECRET": "The webhook secret token did not match.",
    "BAD_PAYLOAD": "The webhook payload was not readable.",
    "EMPTY": "The webhook carried no update.",
}


@dataclass(slots=True)
class NormalizedTelegramMessage:
    update_id: int
    channel_user_id: str
    sender_name: str = ""
    username: str | None = None
    message_type: str = "text"
    text_content: str = ""
    command: str | None = None
    command_args: str | None = None
    callback_query_id: str | None = None
    media_file_id: str | None = None
    media_mime_type: str | None = None
    timestamp_epoch: int = 0
    is_duplicate: bool = False


@dataclass(slots=True)
class TelegramIngressResult:
    is_valid: bool
    status_code: int
    normalized_message: NormalizedTelegramMessage | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


_seen: "OrderedDict[int, float]" = OrderedDict()


def enabled() -> bool:
    return os.getenv("INDOONE_TELEGRAM_INGRESS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset_dedup_cache() -> None:
    _seen.clear()


# ---------------------------------------------------------------- the door


def verify_secret_token(header_token: str | None, expected_secret: str) -> bool:
    """Whether this webhook came from Telegram, compared in constant time.

    ``==`` on a string returns at the first differing byte, which tells an attacker how much
    of the secret was right. ``compare_digest`` does not.
    """
    expected = str(expected_secret or "")
    received = str(header_token or "")

    if not expected or not received:
        return False

    return hmac.compare_digest(received, expected)


# ---------------------------------------------------------------- the window


def _is_duplicate(update_id: int, now: float) -> bool:
    seen_at = _seen.get(update_id)

    if seen_at is not None and now - seen_at <= DEDUP_TTL_SECONDS:
        _seen.move_to_end(update_id)
        return True

    _seen[update_id] = now
    _seen.move_to_end(update_id)

    while _seen:
        oldest, stamp = next(iter(_seen.items()))

        if len(_seen) > DEDUP_MAXSIZE or now - stamp > DEDUP_TTL_SECONDS:
            _seen.popitem(last=False)
            continue

        break

    return False


# ---------------------------------------------------------------- identity


def _sender(user: dict[str, Any]) -> tuple[str, str, str | None]:
    """(channel id, display name, username)."""
    identifier = user.get("id")
    channel = f"tg:{identifier}" if identifier is not None else ""

    first = str(user.get("first_name") or "").strip()
    last = str(user.get("last_name") or "").strip()
    name = " ".join(part for part in (first, last) if part)[:120]

    username = user.get("username")
    return channel, name, (str(username) if username else None)


# ---------------------------------------------------------------- the flattening


def _command_of(text: str) -> tuple[str, str] | None:
    match = _COMMAND.match(text.strip())

    if not match:
        return None

    return match.group("command").casefold(), (match.group("args") or "").strip()


def _media(message: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """(file_id, mime, caption) - the largest size where Telegram sends several."""
    for key in ("voice", "audio", "document", "video", "video_note"):
        payload = message.get(key)

        if isinstance(payload, dict):
            return (
                str(payload.get("file_id") or "") or None,
                str(payload.get("mime_type") or "") or None,
                str(payload.get("caption") or "")[:MAX_TEXT_CHARS],
            )

    photos = message.get("photo")

    if isinstance(photos, list) and photos:
        largest = photos[-1] if isinstance(photos[-1], dict) else {}
        return (
            str(largest.get("file_id") or "") or None,
            "image/jpeg",
            str(message.get("caption") or "")[:MAX_TEXT_CHARS],
        )

    return None, None, ""


def _timestamp(message: dict[str, Any]) -> int:
    try:
        return int(message.get("date") or 0)
    except (TypeError, ValueError):
        return 0


def parse_and_normalize(
    raw_json_str: str, check_duplicate: bool = True, *, now: float | None = None
) -> TelegramIngressResult:
    """Turn one Telegram update into one flat message, or a clean refusal."""
    started = time.perf_counter()

    def done(result: TelegramIngressResult) -> TelegramIngressResult:
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    try:
        payload = json.loads(raw_json_str or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return done(TelegramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["BAD_PAYLOAD"]))

    if not isinstance(payload, dict):
        return done(TelegramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["BAD_PAYLOAD"]))

    try:
        update_id = int(payload.get("update_id") or 0)
    except (TypeError, ValueError):
        update_id = 0

    moment = time.time() if now is None else float(now)

    message = payload.get("message")
    callback = payload.get("callback_query")

    if isinstance(callback, dict):
        user = callback.get("from") if isinstance(callback.get("from"), dict) else {}
        channel, name, username = _sender(user)

        normalized = NormalizedTelegramMessage(
            update_id=update_id,
            channel_user_id=channel,
            sender_name=name,
            username=username,
            message_type="callback_query",
            text_content=str(callback.get("data") or "")[:MAX_TEXT_CHARS],
            callback_query_id=str(callback.get("id") or "") or None,
        )

        if check_duplicate:
            normalized.is_duplicate = _is_duplicate(update_id, moment)

        return done(TelegramIngressResult(is_valid=True, status_code=200, normalized_message=normalized))

    if isinstance(message, dict):
        user = message.get("from") if isinstance(message.get("from"), dict) else {}
        channel, name, username = _sender(user)

        file_id, mime, caption = _media(message)
        text = str(message.get("text") or "")

        if text.strip().startswith("/"):
            command = _command_of(text)
            normalized = NormalizedTelegramMessage(
                update_id=update_id,
                channel_user_id=channel,
                sender_name=name,
                username=username,
                message_type="command",
                text_content=text[:MAX_TEXT_CHARS],
                command=command[0] if command else None,
                command_args=command[1] if command else None,
                timestamp_epoch=_timestamp(message),
            )
        elif file_id:
            normalized = NormalizedTelegramMessage(
                update_id=update_id,
                channel_user_id=channel,
                sender_name=name,
                username=username,
                message_type="media",
                text_content=caption,
                media_file_id=file_id,
                media_mime_type=mime,
                timestamp_epoch=_timestamp(message),
            )
        else:
            normalized = NormalizedTelegramMessage(
                update_id=update_id,
                channel_user_id=channel,
                sender_name=name,
                username=username,
                message_type="text",
                text_content=text[:MAX_TEXT_CHARS],
                timestamp_epoch=_timestamp(message),
            )

        if check_duplicate:
            normalized.is_duplicate = _is_duplicate(update_id, moment)

        return done(TelegramIngressResult(is_valid=True, status_code=200, normalized_message=normalized))

    # A channel post, an edited message, a membership change: Telegram expects a 200 and this
    # is not a user turn. Acknowledged, not refused, and not processed.
    if any(key in payload for key in ("channel_post", "edited_message", "my_chat_member", "edited_channel_post")):
        normalized = NormalizedTelegramMessage(
            update_id=update_id, channel_user_id="", message_type="unsupported"
        )

        if check_duplicate:
            normalized.is_duplicate = _is_duplicate(update_id, moment)

        return done(TelegramIngressResult(is_valid=True, status_code=200, normalized_message=normalized))

    return done(TelegramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["EMPTY"]))
