"""Agent 09 - the Arbiter: what a turn may cost, and what is left to spend.

Every turn used to cost whatever it cost. Two things were missing, and they are different
problems:

* **The bill has no ceiling.** A pasted 50,000-character document is a legitimate-looking
  message that costs more than a day of ordinary use, and nothing stopped it before the
  provider was called.
* **The estimate was wrong in one direction.** Four characters per token is close for
  English and badly low for Kannada: a Devanagari or Kannada glyph cluster is several UTF-8
  bytes and often more than one token. Undercounting is the dangerous direction - it lets a
  prompt past the ceiling that should have been refused.

So this estimates honestly per script, refuses what cannot be afforded, and records what was
spent. It does not decide *what* the answer is worth beyond a ceiling; that is the model's
job and its own output limit.

Three decisions worth stating:

* **It counts tokens, not requests.** ``limits.py`` already counts requests - a burst guard
  and a daily request quota. Deleting that to centralise here would trade a working
  admission control for a token ledger, so the two stay: one guards tempo, this one guards
  cost.
* **It does not cap the provider's output.** ``_generation_config`` in the Gemini service
  deliberately omits ``maxOutputTokens`` so the model uses its own limit; a 256-token cap on
  a question that turns out to need a list would truncate a correct answer to save pennies.
  The allocation is computed and recorded, and it is applied where a cap is already safe -
  the local model path. The ceiling that does bite is on the *input*, before any spend.
* **The quota is per user, rolling, bounded.** 4096 users in memory, oldest evicted. A
  restart forgets the ledger; that is a deliberate choice for an in-process guard, and it is
  why the ceiling on a single prompt is the real defence.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

MAX_TRACKED_USERS = 4096
WINDOW_SECONDS = 24 * 60 * 60.0

# Characters per token, by script. English prose and code sit near 4; Indic scripts carry
# more bytes and more tokens per character, so the divisor is smaller.
_LATIN_CHARS_PER_TOKEN = 4.0
_INDIC_CHARS_PER_TOKEN = 1.8
_CJK_CHARS_PER_TOKEN = 1.5

_INDIC = re.compile(r"[\u0900-\u0dff\u0a80-\u0aff]")
_CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

# Output budgets by kind of question. A factoid answered in a paragraph does not need a
# chapter, and a design document does not fit in a tweet.
FACTOID_TOKENS = 256
TECHNICAL_TOKENS = 1536
ARCHITECTURE_TOKENS = 2048
FAST_PATH_TOKENS = 0

_PROMPT_CEILING_TOKENS = 16_000
_DEFAULT_DAILY_TOKENS = 200_000

# What a refusal says, in the languages this product speaks.
_QUOTA_MESSAGES = {
    "kn": "ಇವತ್ತಿನ ಮಿತಿ ಮುಗಿದಿದೆ. ನಾಳೆ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "hi": "आज की सीमा पूरी हो गई है। कल फिर कोशिश करें।",
    "ta": "இன்றைய வரம்பு முடிந்தது. நாளை மீண்டும் முயலுங்கள்.",
    "te": "ఈరోజు పరిమితి ముగిసింది. రేపు మళ్లీ ప్రయత్నించండి.",
    "en": "You have reached today's usage limit. Please try again tomorrow.",
}

_CEILING_MESSAGES = {
    "kn": "ಈ ಸಂದೇಶ ತುಂಬಾ ದೊಡ್ಡದಿದೆ. ದಯವಿಟ್ಟು ಚಿಕ್ಕದಾಗಿ ಕಳಿಸಿ.",
    "hi": "यह संदेश बहुत बड़ा है। कृपया इसे छोटा करके भेजें।",
    "ta": "இந்தச் செய்தி மிகப் பெரியது. சுருக்கி அனுப்புங்கள்.",
    "te": "ఈ సందేశం చాలా పెద్దది. దయచేసి చిన్నగా పంపండి.",
    "en": "That message is too large. Please send a shorter one.",
}


@dataclass(slots=True)
class _Ledger:
    tokens: int = 0
    window_started: float = 0.0


@dataclass(slots=True)
class BudgetArbitrationResult:
    """Whether this turn may run, with what budget, and what is left."""

    allowed: bool
    allocated_max_tokens: int
    estimated_prompt_tokens: int
    daily_tokens_remaining: int
    rejection_reason: str | None = None
    client_message: dict[str, str] | None = None
    telemetry_micros: float = 0.0


_ledger: "OrderedDict[str, _Ledger]" = OrderedDict()


def enabled() -> bool:
    return os.getenv("INDOONE_TOKEN_ARBITER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset() -> None:
    _ledger.clear()


def daily_allowance() -> int:
    try:
        return max(1_000, int(os.getenv("INDOONE_DAILY_TOKEN_ALLOWANCE", str(_DEFAULT_DAILY_TOKENS))))
    except ValueError:
        return _DEFAULT_DAILY_TOKENS


def prompt_ceiling() -> int:
    try:
        return max(1_000, int(os.getenv("INDOONE_PROMPT_TOKEN_CEILING", str(_PROMPT_CEILING_TOKENS))))
    except ValueError:
        return _PROMPT_CEILING_TOKENS


# ---------------------------------------------------------------- estimation


def estimate_tokens(text: str) -> int:
    """A token projection that errs high for Indic scripts.

    Four characters per token is right for English and wrong for Kannada, where a syllable
    can be several bytes and more than one token. Counting Indic text as English is the
    mistake that lets a prompt slip past the ceiling.
    """
    body = text or ""

    if not body:
        return 0

    indic = len(_INDIC.findall(body))
    cjk = len(_CJK.findall(body))
    latin = max(0, len(body) - indic - cjk)

    tokens = (
        latin / _LATIN_CHARS_PER_TOKEN
        + indic / _INDIC_CHARS_PER_TOKEN
        + cjk / _CJK_CHARS_PER_TOKEN
    )

    # Never fewer than one token for any non-empty text, and always round up: an estimate
    # that rounds down is an estimate that undercounts.
    return max(1, int(tokens + 0.999))


def estimate_prompt_tokens(prompt: str, condensed_messages: list[dict] | None = None) -> int:
    """The prompt plus everything condensed into it."""
    total = estimate_tokens(prompt)

    for message in condensed_messages or []:
        if isinstance(message, dict):
            total += estimate_tokens(str(message.get("content") or ""))
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            total += estimate_tokens(str(message[1]))

    return total


# ---------------------------------------------------------------- allocation


def allocate_max_tokens(intent: str = "", prompt: str = "", language: str = "") -> int:
    """How much room the answer gets.

    Deliberately generous at the top and never zero for a real question: a wrong cap
    truncates a correct answer, which is worse than the tokens it saves.
    """
    kind = (intent or "").strip().upper()
    text = (prompt or "").casefold()

    if kind in {"FAST_PATH", "GENERAL_CHAT"} and kind == "FAST_PATH":
        return FAST_PATH_TOKENS

    # Architecture and design work first: it is the longest, and a keyword scan for "write"
    # would otherwise catch "write a design document" as code.
    if any(
        word in text
        for word in (
            "architecture", "design", "document", "documentation", "plan", "proposal",
            "spec", "roadmap", "ವಾಸ್ತುಶಿಲ್ಪ", "ವಿನ್ಯಾಸ", "ದಾಖಲೆ",
        )
    ):
        return ARCHITECTURE_TOKENS

    if any(
        word in text
        for word in (
            "code", "function", "debug", "trace", "error", "stack", "implement", "script",
            "sql", "test", "refactor", "ಕೋಡ್", "ದೋಷ", "ಸ್ಕ್ರಿಪ್ಟ್",
        )
    ):
        return TECHNICAL_TOKENS

    if kind in {"LIST", "DIFF", "CHECK_HOLIDAY", "READ_EMAIL"}:
        return FACTOID_TOKENS

    return FACTOID_TOKENS


# ---------------------------------------------------------------- the ledger


def _entry(user_id: str, now: float) -> _Ledger:
    key = user_id or "anonymous"
    ledger = _ledger.get(key)

    if ledger is None:
        ledger = _Ledger(tokens=0, window_started=now)
        _ledger[key] = ledger

        while len(_ledger) > MAX_TRACKED_USERS:
            _ledger.popitem(last=False)
    else:
        _ledger.move_to_end(key)

    if now - ledger.window_started >= WINDOW_SECONDS:
        ledger.tokens = 0
        ledger.window_started = now

    return ledger


def get_user_usage(user_id: str, *, now: float | None = None) -> dict:
    """What this user has spent in the current window."""
    moment = time.monotonic() if now is None else float(now)
    ledger = _entry(user_id, moment)
    allowance = daily_allowance()

    return {
        "user_id": user_id or "anonymous",
        "tokens_used": ledger.tokens,
        "tokens_remaining": max(0, allowance - ledger.tokens),
        "allowance": allowance,
        "window_started": ledger.window_started,
    }


def record_usage(user_id: str, actual_tokens_used: int, *, now: float | None = None) -> int:
    """Add to the ledger. Returns what is left afterwards."""
    moment = time.monotonic() if now is None else float(now)
    ledger = _entry(user_id, moment)
    ledger.tokens += max(0, int(actual_tokens_used))

    return max(0, daily_allowance() - ledger.tokens)


def reset_quota(user_id: str) -> None:
    _ledger.pop(user_id or "anonymous", None)


# ---------------------------------------------------------------- the decision


def _message(kind: str, language: str) -> dict[str, str]:
    table = _QUOTA_MESSAGES if kind == "quota" else _CEILING_MESSAGES
    code = (language or "en").strip().casefold()[:2]

    return {"primary": table.get(code, table["en"]), "en": table["en"], "language": code}


def evaluate(
    user_id: str,
    prompt: str,
    condensed_messages: list[dict] | None = None,
    intent: str = "",
    language: str = "",
    *,
    now: float | None = None,
) -> BudgetArbitrationResult:
    """May this turn run, and with what budget?

    Order matters: the size of the prompt is checked before the quota, because a prompt that
    is too large is refused whether or not the user can afford it.
    """
    started = time.perf_counter()
    moment = time.monotonic() if now is None else float(now)

    if not enabled():
        return BudgetArbitrationResult(
            allowed=True,
            allocated_max_tokens=allocate_max_tokens(intent, prompt, language),
            estimated_prompt_tokens=estimate_prompt_tokens(prompt, condensed_messages),
            daily_tokens_remaining=daily_allowance(),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    estimated = estimate_prompt_tokens(prompt, condensed_messages)
    ledger = _entry(user_id, moment)
    remaining = max(0, daily_allowance() - ledger.tokens)
    allocated = allocate_max_tokens(intent, prompt, language)

    if estimated > prompt_ceiling():
        return BudgetArbitrationResult(
            allowed=False,
            allocated_max_tokens=0,
            estimated_prompt_tokens=estimated,
            daily_tokens_remaining=remaining,
            rejection_reason="PROMPT_TOO_LARGE",
            client_message=_message("ceiling", language),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    if estimated > remaining:
        return BudgetArbitrationResult(
            allowed=False,
            allocated_max_tokens=0,
            estimated_prompt_tokens=estimated,
            daily_tokens_remaining=remaining,
            rejection_reason="QUOTA_EXHAUSTED",
            client_message=_message("quota", language),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    return BudgetArbitrationResult(
        allowed=True,
        allocated_max_tokens=allocated,
        estimated_prompt_tokens=estimated,
        daily_tokens_remaining=remaining - estimated,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
