"""Agent 18 - the WhatsApp door: prove it came from Meta, then flatten it.

Meta's webhook payload is a nest four levels deep (``entry[0].changes[0].value.messages[0]``)
with a different shape for a text, a button press, a list choice and a photo. Handing that
to the rest of the system means every consumer learns the nest, and the day Meta adds a field
the whole pipeline learns it again.

So this flattens it once, and does two things the nest cannot do for itself:

* **It proves the message came from Meta.** ``X-Hub-Signature-256`` is an HMAC over the raw
  body, compared in constant time. Without that check anyone who learns the URL can post
  messages as any phone number.
* **It drops the retry.** Meta re-sends a webhook when it does not hear a reply in time, and
  a re-sent message must not become a second answer. A bounded window of recent ``wamid``s
  decides - bounded, because a set that grows forever is a leak, and time-limited, because a
  retry that arrives an hour later is not a retry.

The signature check itself is the one that already exists in the WhatsApp client; this adds
the flattening, the identity and the window, so there is one implementation of each.
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

_DIGITS = re.compile(r"\D+")

_MESSAGES = {
    "BAD_SIGNATURE": "The webhook signature did not match.",
    "BAD_PAYLOAD": "The webhook payload was not readable.",
    "EMPTY": "The webhook carried no message.",
}


@dataclass(slots=True)
class NormalizedWhatsAppMessage:
    wamid: str
    channel_user_id: str
    sender_name: str = ""
    message_type: str = "text"
    text_content: str = ""
    action_id: str | None = None
    media_id: str | None = None
    media_mime_type: str | None = None
    timestamp_epoch: int = 0
    is_duplicate: bool = False


@dataclass(slots=True)
class IngressValidationResult:
    is_valid: bool
    status_code: int
    normalized_message: NormalizedWhatsAppMessage | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


_seen: "OrderedDict[str, float]" = OrderedDict()


def enabled() -> bool:
    return os.getenv("INDOONE_WHATSAPP_INGRESS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset_dedup_cache() -> None:
    _seen.clear()


# ---------------------------------------------------------------- the door


def verify_webhook_signature(
    raw_body: bytes, signature_header: str | None, app_secret: str
) -> bool:
    """Whether this body was signed by Meta with our app secret.

    Delegated to the client that already implements it, so there is one HMAC in the codebase
    rather than two that can disagree about the header format.
    """
    if not app_secret or not str(app_secret).strip():
        return False

    try:
        from app.channels.whatsapp.service import validate_signature
    except Exception:  # a build without the channel still gets a correct check
        import hashlib
        import hmac

        header = signature_header or ""

        if not header.startswith("sha256="):
            return False

        expected = hmac.new(str(app_secret).encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, header.split("=", 1)[1])

    return validate_signature(str(app_secret), signature_header, raw_body)


# ---------------------------------------------------------------- identity


def canonical_channel_id(phone: str) -> str:
    """``+91 98765-43210`` and ``919876543210`` are one person, and must be one id."""
    digits = _DIGITS.sub("", phone or "")

    if not digits:
        return ""

    # An Indian number typed without a country code is still an Indian number.
    if len(digits) == 10:
        digits = "91" + digits

    return f"wa:{digits}"


# ---------------------------------------------------------------- the window


def _is_duplicate(wamid: str, now: float) -> bool:
    if not wamid:
        return False

    seen_at = _seen.get(wamid)

    if seen_at is not None and now - seen_at <= DEDUP_TTL_SECONDS:
        # Refresh the position: a retry proves the id is still in flight.
        _seen.move_to_end(wamid)
        return True

    _seen[wamid] = now
    _seen.move_to_end(wamid)

    while _seen:
        oldest, stamp = next(iter(_seen.items()))

        if len(_seen) > DEDUP_MAXSIZE or now - stamp > DEDUP_TTL_SECONDS:
            _seen.popitem(last=False)
            continue

        break

    return False


# ---------------------------------------------------------------- the flattening


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _message_value(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _first_dict(payload.get("entry"))
    change = _first_dict(entry.get("changes"))
    value = change.get("value")
    return value if isinstance(value, dict) else {}


def _resolve(message: dict[str, Any]) -> tuple[str, str, str | None, str | None, str | None]:
    """(type, text, action_id, media_id, mime) for one message object."""
    kind = str(message.get("type") or "").strip()

    if kind == "text":
        body = message.get("text")
        text = str(body.get("body") or "") if isinstance(body, dict) else ""
        return "text", text[:MAX_TEXT_CHARS], None, None, None

    if kind == "interactive":
        interactive = message.get("interactive")
        interactive = interactive if isinstance(interactive, dict) else {}

        button = interactive.get("button_reply")
        if isinstance(button, dict):
            return (
                "interactive_button",
                str(button.get("title") or "")[:MAX_TEXT_CHARS],
                str(button.get("id") or "") or None,
                None,
                None,
            )

        listing = interactive.get("list_reply")
        if isinstance(listing, dict):
            return (
                "interactive_list",
                str(listing.get("title") or "")[:MAX_TEXT_CHARS],
                str(listing.get("id") or "") or None,
                None,
                None,
            )

        return "interactive_button", "", None, None, None

    if kind == "button":
        # The older template-button shape, which still arrives from live numbers.
        button = message.get("button")
        button = button if isinstance(button, dict) else {}
        return (
            "interactive_button",
            str(button.get("text") or "")[:MAX_TEXT_CHARS],
            str(button.get("payload") or "") or None,
            None,
            None,
        )

    if kind in {"audio", "image", "document", "video", "sticker"}:
        media = message.get(kind)
        media = media if isinstance(media, dict) else {}
        return (
            "media",
            str(media.get("caption") or "")[:MAX_TEXT_CHARS],
            None,
            str(media.get("id") or "") or None,
            str(media.get("mime_type") or "") or None,
        )

    return kind or "unknown", "", None, None, None


def parse_and_normalize(
    raw_json_str: str, check_duplicate: bool = True, *, now: float | None = None
) -> IngressValidationResult:
    """Turn a webhook body into one flat message, or a clean refusal."""
    started = time.perf_counter()

    try:
        payload = json.loads(raw_json_str or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return IngressValidationResult(
            is_valid=False,
            status_code=400,
            error_message=_MESSAGES["BAD_PAYLOAD"],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    if not isinstance(payload, dict):
        return IngressValidationResult(
            is_valid=False,
            status_code=400,
            error_message=_MESSAGES["BAD_PAYLOAD"],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    value = _message_value(payload)
    moment = time.time() if now is None else float(now)

    # A delivery receipt is not a message, and it is not an error either: Meta expects a 200
    # and sends more of them than it sends messages.
    if not value.get("messages"):
        if value.get("statuses"):
            return IngressValidationResult(
                is_valid=True,
                status_code=200,
                normalized_message=NormalizedWhatsAppMessage(
                    wamid="", channel_user_id="", message_type="status", text_content=""
                ),
                telemetry_micros=(time.perf_counter() - started) * 1_000_000,
            )

        return IngressValidationResult(
            is_valid=False,
            status_code=400,
            error_message=_MESSAGES["EMPTY"],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    message = _first_dict(value.get("messages"))

    if not message:
        return IngressValidationResult(
            is_valid=False,
            status_code=400,
            error_message=_MESSAGES["EMPTY"],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    contact = _first_dict(value.get("contacts"))
    profile = contact.get("profile") if isinstance(contact.get("profile"), dict) else {}

    kind, text, action_id, media_id, mime = _resolve(message)
    wamid = str(message.get("id") or "")

    try:
        timestamp = int(str(message.get("timestamp") or "0"))
    except ValueError:
        timestamp = 0

    normalized = NormalizedWhatsAppMessage(
        wamid=wamid,
        channel_user_id=canonical_channel_id(str(message.get("from") or contact.get("wa_id") or "")),
        sender_name=str(profile.get("name") or "")[:120],
        message_type=kind,
        text_content=text,
        action_id=action_id,
        media_id=media_id,
        media_mime_type=mime,
        timestamp_epoch=timestamp,
    )

    if check_duplicate:
        normalized.is_duplicate = _is_duplicate(wamid, moment)

    return IngressValidationResult(
        is_valid=True,
        status_code=200,
        normalized_message=normalized,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
