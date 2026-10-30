"""Agent 22 - the quota guard: a free tier is a budget, and a budget needs a ceiling.

Meta gives a thousand free service conversations a month and bills after that. A channel
that is popular is therefore a channel that costs money, and the worst version of that
surprise arrives as an invoice rather than as an error.

So the ceiling is set at **900**, a hundred conversations short of the free tier, because a
counter that stops exactly at the limit stops one invoice too late: a burst between the check
and the write is enough to cross it.

Two decisions worth stating:

* **The channel decides, not the message.** ``wa:`` is capped; ``tg:`` and ``ig:`` are not,
  because they are free and because they are how new users arrive. The block is a redirect -
  "download the app" - rather than a refusal, so a capped month still converts.
* **The context is RAM only, and it expires.** Fifteen minutes and five turns is enough to
  resolve "show its diff"; a database write for a conversation nobody returns to is cost
  without a reader. The durable store is untouched: this is a buffer in front of it, not a
  replacement for it.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Optional

try:
    from zoneinfo import ZoneInfo

    _KOLKATA: Any = ZoneInfo("Asia/Kolkata")
except Exception:  # a system without tzdata still rolls over on UTC
    _KOLKATA = timezone.utc

WHATSAPP_MONTHLY_LIMIT = 900
MAX_SESSIONS = 2048
MAX_TURNS = 5
SESSION_TTL_SECONDS = 900.0

WA = "whatsapp"
TG = "telegram"
IG = "instagram"
APP = "app"

_CAPPED = {WA}
_PREFIX = {"wa:": WA, "tg:": TG, "ig:": IG}

_APP_CHIP = [{"label": "Download Indoone App", "action": "open_app_download"}]

_EXHAUSTED = {
    "kn": "ಈ ತಿಂಗಳ ಉಚಿತ WhatsApp ಮಿತಿ (900) ಮುಗಿದಿದೆ. ಅನಿಯಮಿತವಾಗಿ ಬಳಸಲು ನಮ್ಮ Indoone ಆ್ಯಪ್ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ 🚀",
    "hi": "इस महीने की मुफ़्त WhatsApp सीमा (900) पूरी हो गई है। असीमित उपयोग के लिए हमारा Indoone ऐप डाउनलोड करें 🚀",
    "en": "This month's free WhatsApp limit (900) is used up. Download the Indoone app to keep chatting without limits.",
}


@dataclass(slots=True)
class ChatTurn:
    user_prompt: str
    assistant_response: str
    timestamp_epoch: float = 0.0


@dataclass(slots=True)
class QuotaCheckResult:
    is_allowed: bool
    channel: str
    current_month_usage: int = 0
    monthly_limit: int | None = None
    quota_exhausted_message: str | None = None
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    telemetry_micros: float = 0.0


# The month's conversations: {(year, month): {channel_user_id, ...}}. A set, not a counter,
# because Meta bills a conversation once however many turns it has.
_conversations: dict[tuple[int, int], set[str]] = {}
_sessions: "OrderedDict[str, Deque[ChatTurn]]" = OrderedDict()


def enabled() -> bool:
    return os.getenv("INDOONE_QUOTA_GUARD_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset_state_for_tests() -> None:
    _conversations.clear()
    _sessions.clear()


# ---------------------------------------------------------------- channels


def channel_of(channel_user_id: str) -> str:
    """``wa:919…`` is a WhatsApp conversation; anything else is not capped."""
    identifier = (channel_user_id or "").strip().casefold()

    for prefix, channel in _PREFIX.items():
        if identifier.startswith(prefix):
            return channel

    return APP


def _month(now: datetime | None) -> tuple[int, int]:
    moment = now or datetime.now(_KOLKATA)

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_KOLKATA)

    local = moment.astimezone(_KOLKATA)
    return local.year, local.month


def _usage(month: tuple[int, int]) -> int:
    return len(_conversations.get(month, ()))


# ---------------------------------------------------------------- the ceiling


def check_quota(channel_user_id: str, now_dt: datetime | None = None, *, language: str = "en") -> QuotaCheckResult:
    """Whether this channel may answer, and what to say when it may not."""
    started = time.perf_counter()

    channel = channel_of(channel_user_id)
    month = _month(now_dt)
    usage = _usage(month)

    if channel not in _CAPPED:
        # Telegram and Instagram are free, and they are the funnel.
        return QuotaCheckResult(
            is_allowed=True,
            channel=channel,
            current_month_usage=usage,
            monthly_limit=None,
            action_chips=[],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    if usage >= WHATSAPP_MONTHLY_LIMIT:
        key = (language or "en").strip().casefold()[:2]

        return QuotaCheckResult(
            is_allowed=False,
            channel=channel,
            current_month_usage=usage,
            monthly_limit=WHATSAPP_MONTHLY_LIMIT,
            quota_exhausted_message=_EXHAUSTED.get(key, _EXHAUSTED["en"]),
            action_chips=[dict(chip) for chip in _APP_CHIP],
            error_code="WA_MONTHLY_QUOTA_EXHAUSTED",
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    return QuotaCheckResult(
        is_allowed=True,
        channel=channel,
        current_month_usage=usage,
        monthly_limit=WHATSAPP_MONTHLY_LIMIT,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def record_usage(channel_user_id: str, now_dt: datetime | None = None) -> None:
    """Count this conversation against the month, once."""
    identifier = (channel_user_id or "").strip()

    if not identifier:
        return

    month = _month(now_dt)
    _conversations.setdefault(month, set()).add(identifier)

    # Only the current and previous month are ever consulted; older sets are dead weight.
    for stale in [key for key in _conversations if key < (month[0], month[1] - 1)]:
        _conversations.pop(stale, None)


# ---------------------------------------------------------------- the buffer


def store_turn(channel_user_id: str, user_prompt: str, assistant_response: str, *, now: float | None = None) -> None:
    """Keep the last few turns in RAM, and no more than that."""
    identifier = (channel_user_id or "").strip()

    if not identifier:
        return

    moment = time.time() if now is None else float(now)
    turns = _sessions.get(identifier)

    if turns is None:
        turns = deque(maxlen=MAX_TURNS)
        _sessions[identifier] = turns

    turns.append(ChatTurn(user_prompt, assistant_response, moment))
    _sessions.move_to_end(identifier)

    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def get_recent_history(
    channel_user_id: str, *, now: float | None = None, ttl_seconds: float = SESSION_TTL_SECONDS
) -> list[tuple[str, str]]:
    """The turns still worth remembering, in order, as (role, text)."""
    identifier = (channel_user_id or "").strip()

    if not identifier:
        return []

    moment = time.time() if now is None else float(now)
    turns = _sessions.get(identifier)

    if turns is None:
        return []

    fresh = [turn for turn in turns if moment - turn.timestamp_epoch <= ttl_seconds]

    if not fresh:
        _sessions.pop(identifier, None)
        return []

    if len(fresh) != len(turns):
        _sessions[identifier] = deque(fresh, maxlen=MAX_TURNS)

    history: list[tuple[str, str]] = []

    for turn in fresh:
        if turn.user_prompt:
            history.append(("user", turn.user_prompt))
        if turn.assistant_response:
            history.append(("assistant", turn.assistant_response))

    return history


def prune(now: float | None = None, ttl_seconds: float = SESSION_TTL_SECONDS) -> int:
    """Drop every session that has gone quiet. Returns how many went."""
    moment = time.time() if now is None else float(now)
    stale = [
        identifier
        for identifier, turns in _sessions.items()
        if not turns or moment - turns[-1].timestamp_epoch > ttl_seconds
    ]

    for identifier in stale:
        _sessions.pop(identifier, None)

    return len(stale)
