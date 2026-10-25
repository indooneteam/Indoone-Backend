"""Admission control for the endpoint that spends money.

Two layers, because they protect against two different things.

**A burst limiter**, in process: a token bucket per subject (the user, or the client
address when there is no user). It answers in memory, costs nothing, and stops one
client - or one retry loop in one client - from opening fifty model calls at once.
It is per instance, which is honest: with more than one instance each holds its own
bucket, so the effective burst limit is per instance. That is the right trade for the
job it does, because a burst is by definition short.

**A daily quota**, in the store: a counter per user per day. It survives a restart,
and with Postgres behind it, it is shared by every instance. This is the layer that
bounds what a user can cost us over a day, so it must not live in a process that can
be recycled.

Neither layer replaces the others that already exist: a tool has its own timeout
(``MAX_TOOL_RUN_SECONDS``), the model loop has a round cap, and each provider call
has its own attempt limit. This module is the gate in front of all of them - the one
place where a request is refused before any of that is paid for.

Why it exists: before this, ``/api/chat`` had no admission control at all. Every
message spends provider tokens, and nothing bounded how many messages a single
account could send. That is a cost risk, an availability risk and an abuse risk at
the same time, and it is the one gap that had to close before anything else in the
production plan could be trusted.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.storage import database
from app.storage.migrations import ensure_schema

# Per subject, per minute, refilled continuously. Thirty a minute is a fast human
# typing with a couple of retries; it is not a batch job.
_DEFAULT_PER_MINUTE = 30.0
# How many requests may arrive back to back before the rate matters. Ten is one
# screenful of taps; a genuine conversation never exceeds it.
_DEFAULT_BURST = 10
# A day's worth of chat for one account. Five hundred messages is far more than a
# person types, and it is the number that bounds what one account can cost.
_DEFAULT_DAILY_QUOTA = 500

# The variable names are written at each call site rather than kept in constants on
# purpose: tests/ops/test_env_documentation.py scans the source for a literal name
# passed to the environment lookup and fails the build when one is not in
# .env.example. A constant would hide a new variable from that guard.

RATE_LIMITED = "RATE_LIMITED"
DAILY_QUOTA_EXCEEDED = "DAILY_QUOTA_EXCEEDED"


@dataclass(frozen=True)
class Admission:
    """The answer to "may this request proceed, and if not, why".

    ``retry_after`` is in whole seconds and is what the client is told to wait. It is
    never zero when the request is refused, so a caller can always set the header.
    """

    allowed: bool
    code: str = ""
    retry_after: int = 0
    message: str = ""
    remaining_today: int | None = None


_ALLOWED = Admission(True)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def limits_enabled() -> bool:
    """Whether admission control is on. On unless explicitly switched off.

    A deployment that has not set the variable gets the protection: defaulting to off
    would mean every existing deployment silently keeps the gap this module closes.
    """

    raw = os.getenv("INDOONE_CHAT_LIMITS_ENABLED", "").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def _per_minute() -> float:
    return _env_float("INDOONE_CHAT_RATE_PER_MINUTE", _DEFAULT_PER_MINUTE)


def _burst() -> int:
    return max(1, _env_int("INDOONE_CHAT_RATE_BURST", _DEFAULT_BURST))


def daily_quota() -> int:
    return max(1, _env_int("INDOONE_CHAT_DAILY_QUOTA", _DEFAULT_DAILY_QUOTA))


# ---- the burst limiter ----------------------------------------------------- #


class _Bucket:
    """One token bucket: tokens refill continuously, each request takes one."""

    __slots__ = ("tokens", "updated")

    def __init__(self, tokens: float, updated: float) -> None:
        self.tokens = tokens
        self.updated = updated


_buckets: dict[str, _Bucket] = {}
_buckets_lock = threading.Lock()
# Buckets are removed once they have refilled completely, so a long-running process
# does not accumulate one dict entry per user who ever sent a message. Pruning runs
# on the request path but does nothing until the table is worth walking.
_PRUNE_AT = 4096


def _take_burst(subject: str, *, now: float | None = None) -> Admission:
    """Take one token for ``subject``, or refuse with the seconds until one is free."""

    per_minute = _per_minute()
    burst = _burst()
    refill_per_second = per_minute / 60.0
    moment = time.monotonic() if now is None else now

    with _buckets_lock:
        bucket = _buckets.get(subject)
        if bucket is None:
            bucket = _Bucket(float(burst), moment)
            _buckets[subject] = bucket
        else:
            elapsed = max(0.0, moment - bucket.updated)
            bucket.tokens = min(float(burst), bucket.tokens + elapsed * refill_per_second)
            bucket.updated = moment

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            if len(_buckets) > _PRUNE_AT:
                _prune(moment, refill_per_second, burst)
            return _ALLOWED

        missing = 1.0 - bucket.tokens
        wait = missing / refill_per_second if refill_per_second > 0 else 60.0
        return Admission(
            False,
            code=RATE_LIMITED,
            retry_after=max(1, int(round(wait))),
            message="That is a lot of messages at once. Try again in a moment.",
        )


def _prune(now: float, refill_per_second: float, burst: int) -> None:
    """Drop buckets that have refilled to full: they carry no information.

    Called with ``_buckets_lock`` held.
    """

    full_after = (burst / refill_per_second) if refill_per_second > 0 else 0.0
    stale = [
        subject
        for subject, bucket in _buckets.items()
        if now - bucket.updated > full_after
    ]
    for subject in stale:
        _buckets.pop(subject, None)


def reset_burst_state() -> None:
    """Clear the in-process buckets. For tests, and for a deliberate restart."""

    with _buckets_lock:
        _buckets.clear()


# ---- the daily quota ------------------------------------------------------- #


def _today() -> str:
    """The window key. A UTC day, so every instance agrees on the boundary."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def initialize() -> None:
    """Bring the schema up to date.

    ``usage_counters`` is created by the migrations like every other table. This is
    kept as a named entry point so the startup path reads the same as the other
    stores'.
    """

    ensure_schema()


def _consume_daily(user_id: str, *, window: str | None = None) -> Admission:
    """Add one to today's counter and refuse if that took the user over the quota.

    The increment and the read happen in one transaction so two requests racing each
    other cannot both read the pre-increment value.
    """

    quota = daily_quota()
    key = window or _today()
    stamp = datetime.now(timezone.utc).isoformat()
    with closing(database.connect()) as db:
        db.execute(
            """
            INSERT INTO usage_counters (subject, window_key, used, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT (subject, window_key)
            DO UPDATE SET used = usage_counters.used + 1, updated_at = excluded.updated_at
            """,
            (user_id, key, stamp),
        )
        row = db.execute(
            "SELECT used FROM usage_counters WHERE subject = ? AND window_key = ?",
            (user_id, key),
        ).fetchone()
        db.commit()

    used = int(row[0]) if row is not None else 1
    if used > quota:
        # The counter keeps climbing past the limit so the next request is refused
        # immediately rather than re-testing; it resets with the window.
        return Admission(
            False,
            code=DAILY_QUOTA_EXCEEDED,
            retry_after=_seconds_until_tomorrow(),
            message=(
                "You have reached today's message limit for this account. "
                "It resets at midnight UTC."
            ),
            remaining_today=0,
        )
    return Admission(True, remaining_today=max(0, quota - used))


def _seconds_until_tomorrow() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow.timestamp() + 86400 - now.timestamp())))


def usage_today(user_id: str, *, window: str | None = None) -> int:
    """How many requests this user has made today. For ops, not for the request path."""

    normalized = str(user_id or "").strip()
    if not normalized:
        return 0
    key = window or _today()
    try:
        with closing(database.connect()) as db:
            row = db.execute(
                "SELECT used FROM usage_counters WHERE subject = ? AND window_key = ?",
                (normalized, key),
            ).fetchone()
    except Exception:
        return 0
    return int(row[0]) if row is not None else 0


# ---- the one function the request path calls ------------------------------- #


def admit(user_id: str, *, client_ip: str = "") -> Admission:
    """Decide whether this request may run, and charge it if it may.

    Order matters: the burst check first, because it is free and it is the one that
    protects the process; then the durable quota, because it costs a write and it is
    the one that protects the budget. A request refused for burst does not spend a
    unit of the daily quota - a client that retries too fast should not also lose
    quota for the retries.

    ``client_ip`` is used only when there is no authenticated user, so an
    unauthenticated flood is still bounded by something.
    """

    if not limits_enabled():
        return _ALLOWED

    subject = str(user_id or "").strip()
    if not subject:
        address = str(client_ip or "").strip()
        if not address:
            # Nothing to key on. Refusing here would break a caller that cannot be
            # identified; the route requires a principal before it reaches us, so this
            # is a defensive branch, not a policy.
            return _ALLOWED
        return _take_burst(f"ip:{address}")

    burst = _take_burst(f"user:{subject}")
    if not burst.allowed:
        return burst

    try:
        return _consume_daily(subject)
    except Exception:
        # A store that cannot answer must not lock every user out of the product. The
        # burst limiter has already run, so the request is still bounded in rate; the
        # quota is the layer that is skipped, and that is the safer failure.
        return _ALLOWED
