"""How much of the conversation the model is shown, and how to ask for more.

Every regular turn used to carry the last twelve messages. Most of that was noise: the
question is usually about what was said a moment ago, and the rest is paid for on every
single request.

The model now sees four messages - two exchanges. The full thread is still stored, and a
question about earlier in the conversation is answered from a digest, on demand, without a
model call.

The window applies to the MODEL's payload only. The routers that bind ordinals and
repository names read the wider history, because a list shown six messages ago is still
what "2" refers to.
"""

from __future__ import annotations

# Two exchanges: the message being answered, and the one before it.
DEFAULT_HISTORY_LIMIT = 4


def history_window(
    history: list[tuple[str, str]] | None, limit: int = DEFAULT_HISTORY_LIMIT
) -> list[tuple[str, str]]:
    """The last ``limit`` messages, in order. Never more."""
    if not history:
        return []

    return list(history[-max(1, limit):])
