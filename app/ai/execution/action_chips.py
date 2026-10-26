"""Clickable follow-ups attached to a reply.

A held write answered with "reply CONFIRM to go ahead, or CANCEL to stop" - the user had
to type a word to press a button that did not exist. The data needed to render it was
already in the reply; what was missing was a field to carry it to the client.

A ContextVar rather than a parameter: the chips are produced deep inside a tool executor
and consumed at the top of the request handler, and threading a mutable list through
every signature in between would touch far more code than the feature is worth. The
variable is set once per request, so two concurrent requests cannot see each other's
chips.
"""

from __future__ import annotations

from contextvars import ContextVar

# The chips for the request in flight. None means nobody asked for them - an executor
# called from a worker thread or a test does not pay for this.
_current: ContextVar[list[dict[str, str]] | None] = ContextVar("action_chips", default=None)

# The two answers a held write understands. The action is the exact text the chat
# endpoint resolves, so pressing the button sends what typing would have sent.
APPROVE = {"label": "Confirm", "action": "CONFIRM"}
CANCEL = {"label": "Cancel", "action": "CANCEL"}


def begin() -> None:
    """Start collecting for this request."""
    _current.set([])


def emit(*chips: dict[str, str]) -> None:
    """Add chips for the current request. A no-op when nobody is collecting."""
    bucket = _current.get()

    if bucket is None:
        return

    for chip in chips:
        label = str(chip.get("label") or "").strip()
        action = str(chip.get("action") or "").strip()

        if not label or not action:
            continue

        candidate = {"label": label, "action": action}

        if candidate not in bucket:
            bucket.append(candidate)


def collected() -> list[dict[str, str]]:
    """What was collected for the current request."""
    return [dict(chip) for chip in (_current.get() or [])]


def reset() -> None:
    """Stop collecting. Called at the end of a request so nothing leaks to the next one."""
    _current.set(None)
