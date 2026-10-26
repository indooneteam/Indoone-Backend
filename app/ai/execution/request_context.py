"""The two lines every model call needs, wherever the message came from.

A message can arrive from the Android app, the web app, WhatsApp, Telegram or
Instagram. The instruction that goes with it should not depend on which: the same
user, the same clock, the same connected accounts.

Both lines used to be built inside `app/api/chat/chat.py`, which was fine while the
chat endpoint was the only caller. The channel webhooks called the model directly, so
a message on WhatsApp was answered with no date and no statement about the user's
accounts - and a model with no date answers a date question from its training memory,
and a model with no statement about connected accounts answers "I don't have access".

Keeping one copy here is what stops the two paths from drifting again.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.capabilities.connectors.store import list_memories

_IST = timezone(timedelta(hours=5, minutes=30))


def user_name_line(user_id: str) -> str:
    """A one-line instruction naming the user, or "" when the name is unknown.

    The name lives in the memory store, so it is read fresh on every request. Only the
    name is sent: no other stored detail reaches the model.
    """

    normalized = str(user_id or "").strip()
    if not normalized:
        return ""
    try:
        rows = list_memories(normalized, limit=20)
    except Exception:
        return ""
    for wanted in ("name", "nickname"):
        for row in rows:
            if str(row.get("key") or "").strip().casefold() != wanted:
                continue
            value = " ".join(str(row.get("value") or "").split()).strip()
            if value:
                return f"The user's name is {value[:60]}."
    return ""


def now_line() -> str:
    """One short line giving today's date and time in IST.

    Without it the model answers a date question from its training memory, which is
    years out of date. The line is built per request from the server clock.
    """

    now = datetime.now(_IST)
    return f"Today is {now:%Y-%m-%d} ({now:%A}), {now:%H:%M} IST."
