"""Agent 07 - the Treasury: an answer already given, given again for nothing.

A cache is a promise that two questions are the same question. That promise is easy to break
in three ways, and this agent refuses all three:

* **Private data.** "unread emails", "git diff", "my reminders" are different every time.
  A cache that serves them is a data leak with extra steps, so anything about the mailbox,
  a repository or the thread is never stored - the subject decides, not the wording.
* **Answers that move.** "naale raja na?" is answered by a date, and a date that is right
  today is wrong tomorrow. Relative questions are refused even though they look static,
  which is why the canonical key can still fold "naale" and "nale" together - the folding is
  correct, the storing is not.
* **Actions.** "merge the pr", "send the mail" change state. Storing a success would tell
  the next person it already happened.

What is left is what a treasury is for: public facts, the gazetted calendar, documentation,
and the explanations that do not change between two askers.

The key is built from the cleaned query, the language and the register, because the same
question in Kannada and in English deserves two different answers. Spelling variants of the
same word fold to one key - "naale" and "nale" are one word spelled twice - and the fold is
limited to a known vocabulary, so an unknown word is never guessed at.

Bounded at 2048 entries, LRU-evicted, TTL per kind (a day for the calendar, six hours for
documentation), standard library only, and a probe that costs a dictionary lookup.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

MAX_ENTRIES = 2048
HOLIDAY_TTL_SECONDS = 24 * 60 * 60
GENERAL_TTL_SECONDS = 6 * 60 * 60
PROBE_BUDGET_MILLISECONDS = 0.05

_WORD = re.compile(r"[a-z0-9]+|[\u0900-\u0dff]+")

# Spelling variants of one word. Folding is limited to this vocabulary on purpose: guessing
# that two unknown words are the same word is how a cache starts answering the wrong thing.
_FOLD: dict[str, str] = {
    "naale": "tomorrow", "nale": "tomorrow", "naaley": "tomorrow", "ನಾಳೆ": "tomorrow",
    "ivattu": "today", "ivathu": "today", "indu": "today", "ಇಂದು": "today",
    "raja": "rajyotsava", "rajyotsava": "rajyotsava", "ರಾಜ": "rajyotsava",
    "ರಾಜ್ಯೋತ್ಸವ": "rajyotsava",
    "habba": "holiday", "habbha": "holiday", "ಹಬ್ಬ": "holiday", "holiday": "holiday",
    "holidays": "holiday", "raje": "holiday", "ರಜೆ": "holiday", "chutti": "holiday",
    "deepavali": "deepavali", "diwali": "deepavali", "ದೀಪಾವಳಿ": "deepavali",
    "ugadi": "ugadi", "ಯುಗಾದಿ": "ugadi",
    "dusra": "second", "doosra": "second", "second": "second",
}

# A question whose answer is a date that moves. Never stored, however static it looks.
_RELATIVE = re.compile(
    r"\b(today|tomorrow|tonight|this\s+week|this\s+month|next\s+week|now|currently|latest|"
    r"naale|nale|ivattu|ivathu|inda|e\s*vara|ee\s*vara|e\s*tingalu)\b"
    r"|ಇಂದು|ನಾಳೆ|ಈ\s*ವಾರ|ಈ\s*ತಿಂಗಳು|ಇವತ್ತು",
    re.IGNORECASE,
)

# Anything that changes state. A stored success would be a lie to the next asker.
_MUTATION = re.compile(
    r"\b(send|sent|delete|remove|create|make|merge|push|deploy|draft|reply|forward|"
    r"update|edit|rename|close|reopen|approve|revoke|cancel)\b"
    r"|ಕಳಿಸು|ಅಳಿಸು|ಮಾಡು|ರಚಿಸು|ಮರ್ಜ್|ಡಿಪ್ಲಾಯ್",
    re.IGNORECASE,
)

# Subjects that are private or live by nature.
_PRIVATE_SUBJECTS = frozenset({"email", "code", "conversation"})


@dataclass(slots=True)
class _Entry:
    response: str
    action_chips: list[dict] | None
    stored_at: float
    expires_at: float
    tokens: int
    hits: int = 0


@dataclass(slots=True)
class TreasuryCacheResult:
    """A hit, or a miss that says what key it looked for."""

    cache_hit: bool
    cached_response: str | None = None
    action_chips: list[dict] | None = None
    tokens_saved: int = 0
    canonical_key: str = ""
    telemetry_micros: float = 0.0


_entries: "OrderedDict[str, _Entry]" = OrderedDict()
_hits = 0
_misses = 0
_tokens_saved = 0


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_TREASURY_CACHE_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset() -> None:
    """Empty the treasury. Used by tests and by an operator after a bad answer."""
    global _hits, _misses, _tokens_saved

    _entries.clear()
    _hits = 0
    _misses = 0
    _tokens_saved = 0


def stats() -> dict[str, int]:
    return {
        "entries": len(_entries),
        "hits": _hits,
        "misses": _misses,
        "tokens_saved": _tokens_saved,
    }


# ---------------------------------------------------------------- the key


def canonical_key(cleaned_query: str, language: str = "", register: str = "") -> str:
    """One string for one question, however it was spelled.

    Language and register are part of the key: "when is deepavali" and "ದೀಪಾವಳಿ ಯಾವಾಗ" are
    the same fact but not the same answer, and a respectful question deserves a respectful
    reply rather than the informal one that was cached first.
    """
    tokens = [
        _FOLD.get(token, token)
        for token in _WORD.findall((cleaned_query or "").casefold())
    ]

    body = " ".join(tokens)
    return f"{(language or '').strip().casefold()}|{(register or '').strip().casefold()}|{body}"


# ---------------------------------------------------------------- eligibility


def should_store(cleaned_query: str, subject: str = "") -> bool:
    """Whether this question may be remembered at all.

    The three refusals, in order of how much damage they do if wrong: private data, an
    answer that moves, and an action that already happened.
    """
    text = (cleaned_query or "").strip()

    if not text:
        return False

    if subject in _PRIVATE_SUBJECTS:
        return False

    if _RELATIVE.search(text):
        return False

    if _MUTATION.search(text):
        return False

    return True


def ttl_for(cleaned_query: str) -> int:
    """A day for the calendar, six hours for everything else."""
    lowered = (cleaned_query or "").casefold()

    if any(word in lowered for word in ("holiday", "festival", "rajyotsava", "deepavali", "ugadi", "habba", "ರಜೆ", "ಹಬ್ಬ")):
        return HOLIDAY_TTL_SECONDS

    return GENERAL_TTL_SECONDS


# ---------------------------------------------------------------- the treasury


def get(
    cleaned_query: str,
    language: str = "",
    register: str = "",
    *,
    now: float | None = None,
) -> TreasuryCacheResult:
    """Look the question up. A miss costs one dictionary lookup."""
    global _hits, _misses

    started = time.perf_counter()
    moment = time.monotonic() if now is None else float(now)
    key = canonical_key(cleaned_query, language, register)

    entry = _entries.get(key)

    if entry is None:
        _misses += 1
        return TreasuryCacheResult(
            cache_hit=False,
            canonical_key=key,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    if moment >= entry.expires_at:
        # Expired is a miss, and the entry goes: leaving it would only be found again.
        _entries.pop(key, None)
        _misses += 1
        return TreasuryCacheResult(
            cache_hit=False,
            canonical_key=key,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    entry.hits += 1
    _entries.move_to_end(key)
    _hits += 1

    return TreasuryCacheResult(
        cache_hit=True,
        cached_response=entry.response,
        action_chips=entry.action_chips,
        tokens_saved=entry.tokens,
        canonical_key=key,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def put(
    cleaned_query: str,
    language: str = "",
    register: str = "",
    response: str = "",
    action_chips: list[dict] | None = None,
    tokens: int = 0,
    ttl_seconds: int | None = None,
    *,
    subject: str = "",
    now: float | None = None,
) -> bool:
    """Store an answer, if the question is one that may be stored.

    Returns whether it was stored, so a caller can say so rather than guess.
    """
    global _tokens_saved

    if not (response or "").strip():
        return False

    if not should_store(cleaned_query, subject):
        return False

    moment = time.monotonic() if now is None else float(now)
    key = canonical_key(cleaned_query, language, register)
    lifetime = ttl_for(cleaned_query) if ttl_seconds is None else int(ttl_seconds)

    _entries[key] = _Entry(
        response=response,
        action_chips=action_chips,
        stored_at=moment,
        expires_at=moment + lifetime,
        tokens=max(0, int(tokens)),
    )
    _entries.move_to_end(key)
    _tokens_saved += max(0, int(tokens))

    while len(_entries) > MAX_ENTRIES:
        _entries.popitem(last=False)

    return True


def invalidate(canonical: str) -> bool:
    """Forget one answer, by key."""
    return _entries.pop(canonical, None) is not None


def clear() -> None:
    _entries.clear()
