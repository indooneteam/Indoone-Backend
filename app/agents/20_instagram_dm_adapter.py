"""Agent 20 - the Instagram door: a DM is a DM, and it has no markdown.

Instagram sits in the same Meta ecosystem as WhatsApp, and its direct messages are the most
restricted of the three channels: **no markdown at all**, and **one thousand characters**.
Two things follow, and they are the two things this agent exists for.

**Inbound.** The signature is the same HMAC over the raw body, so it is checked in constant
time. Then the graph payload (``entry[0].messaging[0]``) is flattened into one typed message:
a DM, a quick reply, a story mention, or an attachment pointer. A bot's own message comes
back as an **echo** - and an echo that is processed becomes an answer to itself, then an
answer to that answer. It is silenced here, before anything downstream can loop on it.

**Outbound.** The egress arbiter already knows how to render for Instagram, including how to
cut without splitting a Kannada conjunct. This delegates to it rather than growing a second
formatter that can disagree - the only thing added here is the markdown-to-text conversion
that a DM needs and a chat window does not.
"""

from __future__ import annotations

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
INSTAGRAM_LIMIT = 1000

_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")

# Kannada, Devanagari, Telugu, Tamil, Malayalam, Bengali, Gujarati.
_VIRAMAS = frozenset({"\u0ccd", "\u094d", "\u0c4d", "\u0bcd", "\u0d4d", "\u09cd", "\u0acd"})

_MESSAGES = {
    "BAD_SIGNATURE": "The webhook signature did not match.",
    "BAD_PAYLOAD": "The webhook payload was not readable.",
    "EMPTY": "The webhook carried no message.",
}


@dataclass(slots=True)
class NormalizedInstagramMessage:
    mid: str
    channel_user_id: str
    message_type: str = "text"
    text_content: str = ""
    action_payload: str | None = None
    attachment_type: str | None = None
    attachment_url: str | None = None
    timestamp_epoch: int = 0
    is_duplicate: bool = False
    is_echo: bool = False


@dataclass(slots=True)
class InstagramIngressResult:
    is_valid: bool
    status_code: int
    normalized_message: NormalizedInstagramMessage | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


_seen: "OrderedDict[str, float]" = OrderedDict()


def enabled() -> bool:
    return os.getenv("INDOONE_INSTAGRAM_INGRESS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset_dedup_cache() -> None:
    _seen.clear()


# ---------------------------------------------------------------- the door


def verify_signature(raw_body: bytes, signature_header: str | None, app_secret: str) -> bool:
    """The same Meta HMAC as WhatsApp, so the same implementation checks it."""
    if not app_secret or not str(app_secret).strip():
        return False

    try:
        from app.channels.whatsapp.service import validate_signature
    except Exception:
        import hashlib
        import hmac

        header = signature_header or ""

        if not header.startswith("sha256="):
            return False

        expected = hmac.new(str(app_secret).encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, header.split("=", 1)[1])

    return validate_signature(str(app_secret), signature_header, raw_body)


# ---------------------------------------------------------------- the window


def _is_duplicate(mid: str, now: float) -> bool:
    if not mid:
        return False

    seen_at = _seen.get(mid)

    if seen_at is not None and now - seen_at <= DEDUP_TTL_SECONDS:
        _seen.move_to_end(mid)
        return True

    _seen[mid] = now
    _seen.move_to_end(mid)

    while _seen:
        oldest, stamp = next(iter(_seen.items()))

        if len(_seen) > DEDUP_MAXSIZE or now - stamp > DEDUP_TTL_SECONDS:
            _seen.popitem(last=False)
            continue

        break

    return False


# ---------------------------------------------------------------- egress


def format_for_instagram(markdown_text: str) -> str:
    """Plain text, inside a thousand characters, with no conjunct cut in half.

    A DM has no bold and no links, so the markup is removed rather than escaped - and the cut
    is delegated to the egress arbiter, which already knows that a Kannada cluster is not a
    character.
    """
    body = markdown_text or ""

    body = _CODE_FENCE.sub(lambda match: match.group(1), body)
    body = _INLINE_CODE.sub(lambda match: match.group(1), body)
    body = _LINK.sub(lambda match: f"{match.group(1)} ({match.group(2)})", body)
    body = _HEADING.sub("", body)
    body = _BLOCKQUOTE.sub("", body)
    body = _RULE.sub("", body)
    body = _BOLD.sub(lambda match: match.group(1) or match.group(2) or "", body)
    body = _ITALIC.sub(lambda match: match.group(1) or match.group(2) or "", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if len(body) <= INSTAGRAM_LIMIT:
        return body

    try:
        from app.agents import model_arbiter

        rendered = str(model_arbiter().to_instagram(body).get("text") or "")
    except Exception:
        # A build without the arbiter still cuts somewhere safe rather than mid-cluster.
        cut = INSTAGRAM_LIMIT - 1

        while cut > 0 and _is_unsafe_cut(body, cut):
            cut -= 1

        rendered = body[:cut].rstrip() + "…"

    return _trim_dangling_marks(rendered or body[:INSTAGRAM_LIMIT])


def _trim_dangling_marks(text: str) -> str:
    """Drop a trailing virama or combining mark left behind by a word-boundary cut.

    A Tamil word often ends in a virama, so "cut at the last space" can leave the mark
    hanging - which is a broken letter on the screen even though the cut itself was safe.
    """
    import unicodedata

    body = text
    trailing = ""

    if body.endswith("…"):
        body, trailing = body[:-1], "…"

    trimmed = body.rstrip()

    while trimmed and (
        unicodedata.combining(trimmed[-1]) or trimmed[-1] in _VIRAMAS
    ):
        trimmed = trimmed[:-1].rstrip()

    return trimmed + trailing


def _is_unsafe_cut(text: str, index: int) -> bool:
    import unicodedata

    if index <= 0 or index >= len(text):
        return False

    if unicodedata.combining(text[index]):
        return True

    # A virama binds the next consonant to this one; cutting after it orphans the pair.
    return text[index - 1] in _VIRAMAS


# ---------------------------------------------------------------- the flattening


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def parse_and_normalize(
    raw_json_str: str, check_duplicate: bool = True, *, now: float | None = None
) -> InstagramIngressResult:
    """Turn one graph webhook into one flat DM, or a clean refusal."""
    started = time.perf_counter()

    def done(result: InstagramIngressResult) -> InstagramIngressResult:
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    try:
        payload = json.loads(raw_json_str or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return done(InstagramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["BAD_PAYLOAD"]))

    if not isinstance(payload, dict):
        return done(InstagramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["BAD_PAYLOAD"]))

    entry = _first_dict(payload.get("entry"))
    event = _first_dict(entry.get("messaging"))
    message = event.get("message")

    if not isinstance(message, dict):
        # A read receipt or a delivery mark: acknowledged, and not a turn.
        if "read" in event or "delivery" in event or "reaction" in event:
            return done(
                InstagramIngressResult(
                    is_valid=True,
                    status_code=200,
                    normalized_message=NormalizedInstagramMessage(mid="", channel_user_id="", message_type="system"),
                )
            )

        return done(InstagramIngressResult(is_valid=False, status_code=400, error_message=_MESSAGES["EMPTY"]))

    mid = str(message.get("mid") or "")
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    sender_id = str(sender.get("id") or "")
    channel = f"ig:{sender_id}" if sender_id else ""

    try:
        timestamp = int(event.get("timestamp") or message.get("timestamp") or 0) // 1000
    except (TypeError, ValueError):
        timestamp = 0

    # The bot's own message coming back. Processing it means answering yourself, then
    # answering that answer - a loop with no natural end.
    if message.get("is_echo"):
        normalized = NormalizedInstagramMessage(
            mid=mid,
            channel_user_id=channel,
            message_type="echo",
            text_content=str(message.get("text") or "")[:MAX_TEXT_CHARS],
            timestamp_epoch=timestamp,
            is_echo=True,
        )
        return done(InstagramIngressResult(is_valid=True, status_code=200, normalized_message=normalized))

    quick_reply = message.get("quick_reply")
    attachments = message.get("attachments")
    attachment = _first_dict(attachments)

    if isinstance(quick_reply, dict):
        normalized = NormalizedInstagramMessage(
            mid=mid,
            channel_user_id=channel,
            message_type="quick_reply",
            text_content=str(message.get("text") or "")[:MAX_TEXT_CHARS],
            action_payload=str(quick_reply.get("payload") or "") or None,
            timestamp_epoch=timestamp,
        )
    elif attachment:
        normalized = NormalizedInstagramMessage(
            mid=mid,
            channel_user_id=channel,
            message_type="attachment",
            text_content=str(message.get("text") or "")[:MAX_TEXT_CHARS],
            attachment_type=str(attachment.get("type") or "") or None,
            attachment_url=str(attachment.get("url") or "") or None,
            timestamp_epoch=timestamp,
        )
    else:
        normalized = NormalizedInstagramMessage(
            mid=mid,
            channel_user_id=channel,
            message_type="text",
            text_content=str(message.get("text") or "")[:MAX_TEXT_CHARS],
            timestamp_epoch=timestamp,
        )

    if check_duplicate:
        normalized.is_duplicate = _is_duplicate(mid, time.time() if now is None else float(now))

    return done(InstagramIngressResult(is_valid=True, status_code=200, normalized_message=normalized))
