"""A promise is not an answer.

Seen in the app: "One moment, I will fetch the latest news" - and nothing was fetched. The
user had to send "kodu" to get the answer the first message should have carried.

Two things fix that. The research fast path runs the search before the model speaks, so
there is usually nothing left to promise. This module is the second half: whatever reaches
the user as a final reply is checked, and a promise is replaced by the work it promised.
"""

from __future__ import annotations

import re

# Promise shapes, in the three ways people type them. Deliberately narrow: these are the
# phrases that mean "not now", never the phrase that means "here it is".
_PROMISE = re.compile(
    r"\b(one|just a|a) moment\b"
    r"|\bhold (on|tight)\b"
    r"|\b(i|i'll|i will|let me)\s+(will\s+)?(fetch|search|look|check|find|get|gather|collect)\b"
    r"|\bi(?:'ll| will)\s+(fetch|search|look|check|find|get|gather|collect)\b"
    r"|\b(fetching|searching|looking\s+it\s+up|gathering)\b"
    r"|\bwait\b"
    r"|ಒಂದು\s*ಕ್ಷಣ|ಕ್ಷಣ\s*ಕಾಯಿರಿ|ತರ್ತೀನಿ|ತರುತ್ತೇನೆ|ಹುಡುಕಿ\s*ತರುತ್ತೇನೆ|ನೋಡಿ\s*ಹೇಳುತ್ತೇನೆ",
    re.IGNORECASE,
)

# A long reply that mentions a lookup in passing is not a promise; a short one that says
# nothing else is.
_MAX_PROMISE_WORDS = 45


def is_promise(reply: str) -> bool:
    """True when a reply promises a lookup instead of answering."""
    cleaned = (reply or "").strip()

    if not cleaned:
        return False

    if len(cleaned.split()) > _MAX_PROMISE_WORDS:
        return False

    return bool(_PROMISE.search(cleaned))
