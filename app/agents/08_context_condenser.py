"""Agent 08 - the Condenser: send the context the turn needs, and no more.

Every turn used to be handed the last few messages whether it needed them or not. That is
expensive in the easy case and unhelpful in the hard one: "what is docker" pays for ten turns
of chat it will ignore, while "show its diff" is handed the same ten turns and left to work
out which "it" was meant.

This decides instead, in three steps:

1. **Does this turn need history at all?** A self-contained question - a public fact, a
   direct command, a calendar question - is answered alone. The history goes, and with it
   every token it would have cost.
2. **If it does, what does it refer to?** A pronoun in any of the languages this product
   serves ("adara", "uska", "adhoda", "its", "daani") means the turn is riding on the one
   before it. The target is read out of that turn - a branch, a commit, a pull request, a
   file - and reported, so the caller knows what "it" was.
3. **How much of the rest is worth sending?** The last two turns go whole; everything older
   becomes one line of state. A conversation is not a transcript to replay, it is a
   situation to describe.

Standard library only, bounded, and a decision that costs a scan of the message.

What this deliberately does not replace: ``conversation_digest``. That module answers a user
who asks "what did we discuss" - a visible feature. This one decides what the model is told,
which is invisible. Merging them would tie a router to a prompt budget.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

INDEPENDENT = "INDEPENDENT"
ANAPHORA_RESOLVED = "ANAPHORA_RESOLVED"
SEQUENTIAL_FOLLOWUP = "SEQUENTIAL_FOLLOWUP"

# How many turns are kept whole, and how many older ones feed the state line.
FULL_TURNS = 2
DIGEST_TURNS = 8
MAX_DIGEST_CHARS = 240
PROBE_BUDGET_MILLISECONDS = 0.05

# Pronouns and demonstratives that mean "the thing we were just talking about".
_ANAPHORA = re.compile(
    r"\b(its|it|that|those|them|their|this|previous|above|same|former|latter)\b"
    r"|\b(adara|adannu|adakke|alli|avara|avaru|adhu|adu|matte|innu)\b"
    r"|\b(uska|usme|usko|uska|iske|unka|wahan|usse|uski|aur\s+batao|phir\s+kya)\b"
    r"|\b(adhoda|adhai|adhu|daani|dantlo|malli|inkenti)\b"
    r"|ಅದರ|ಅದನ್ನು|ಅದಕ್ಕೆ|ಅಲ್ಲಿ|ಅವರ|ಅದು|ಮತ್ತೆ|ಇನ್ನೂ"
    r"|ಅದರಲ್ಲಿ|ಅದೇ|ಇದರ|ಇದನ್ನು|ಇದರಲ್ಲಿ|ಇದಕ್ಕೆ"
    r"|उसका|उसकी|उसके|उसमें|उसको|इसका|इसको|इसकी|वहाँ|उससे|फिर\s*क्या|और\s*बताओ"
    r"|அதன்|அதை|அதில்|இதன்|இதை|இதில்|அங்கே|மேலும்"
    r"|దాని|దానిలో|దీని|దీంట్లో|అక్కడ|మళ్లీ|ఇంకేంటి|అందులో"
    r"|তার|সেটা|ওটা|ওখানে|আরও\s*বলো"
    r"|അതിന്റെ|അത്|ഇതിന്റെ|ഇത്|വീണ്ടും"
    r"|તેનું|એ|ત્યાં|વધુ\s*કહો"
    r"|ಅಲ್ಲಿ\s*ಏನು|हुआ\s*क्या",
    re.IGNORECASE,
)

# A follow-up with no pronoun but no subject either: "and then?", "aur batao", "ಮತ್ತೆ?"
_FOLLOWUP = re.compile(
    r"^\s*(and|then|so|ok|okay|continue|go\s+on|more|next)\b"
    r"|\b(aur\s+batao|phir\s+kya|matte\s+helu|innu\s+helu|inkenti)\b"
    r"|ಮತ್ತೆ\s*ಹೇಳು|ಇನ್ನೂ\s*ಹೇಳು|आगे\s*बताओ",
    re.IGNORECASE,
)

# Things a pronoun can point at. One pattern with named groups rather than five patterns:
# five searches per message over a ten-message thread is most of the budget, and one is not.
_ENTITY = re.compile(
    r"(?P<branch>\b(?:branch|ಶಾಖೆ)\s+(?P<branch_name>[A-Za-z0-9._/-]{2,60}))"
    r"|(?P<pull_request>\b(?:pr|pull\s+request)\s*#?(?P<pr_number>\d{1,6})\b)"
    r"|(?P<commit>\b(?P<sha>[0-9a-f]{7,40})\b)"
    r"|(?P<file>\b(?P<path>[\w./-]+\.(?:py|kt|js|ts|md|json|ya?ml|sql))\b)",
    re.IGNORECASE,
)

# A question about the conversation itself. No pronoun, but history is the subject.
_HISTORY_QUESTION = re.compile(
    r"\b(what\s+did\s+(i|we)|what\s+have\s+(i|we)|did\s+i|earlier|previous|"
    r"last\s+(message|question|answer|time|turn)|remember|so\s+far|summary)\b"
    r"|ಹಿಂದೆ|ಮೊದಲು|ಮೊದಲಿನ|आधी|पहले|ಸಾರಾಂಶ",
    re.IGNORECASE,
)

# The shape of a question that carries its own subject.
_STANDALONE_QUESTION = re.compile(
    r"^\s*(what|who|how|why|when|where|which|explain|define|tell\s+me\s+about)\b"
    r"|^\s*(ಏನು|ಯಾವಾಗ|ಹೇಗೆ|ಎಲ್ಲಿ|ಯಾರು|ವಿವರಿಸಿ)"
    r"|^\s*(kya|kaise|kab|kahan|kaun)\b",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|namaskara|namaste|thanks|thank\s+you|ok|okay|dhanyavada)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CondensedContextResult:
    """What the model should be told, and why."""

    requires_history: bool
    condensed_messages: list[dict[str, str]] = field(default_factory=list)
    resolved_target: str | None = None
    dependency_type: str = INDEPENDENT
    tokens_saved_estimate: int = 0
    telemetry_micros: float = 0.0


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_CONTEXT_CONDENSER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _estimate_tokens(text: str) -> int:
    """Four characters a token is the usual rule of thumb, and it is close enough here."""
    return max(0, len(text) // 4)


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    if isinstance(message, (list, tuple)) and len(message) == 2:
        return str(message[1])
    return str(message or "")


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "user")
    if isinstance(message, (list, tuple)) and len(message) == 2:
        return str(message[0])
    return "user"


def resolve_target(history: list[object]) -> str | None:
    """The most specific thing the last few turns were about.

    Read backwards, because the nearest mention is the one a pronoun means.
    """
    for message in reversed(list(history)):
        found = _ENTITY.search(_message_text(message))

        if found is None:
            continue

        if found.group("branch_name"):
            return f"branch:{found.group('branch_name')}"
        if found.group("pr_number"):
            return f"pull_request:{found.group('pr_number')}"
        if found.group("sha"):
            return f"commit:{found.group('sha')}"
        if found.group("path"):
            return f"file:{found.group('path')}"

    return None


def _digest_line(history: list[object]) -> str:
    """One line describing what the older turns were about."""
    topics: list[str] = []
    scanned = 0

    for message in history:
        text = " ".join(_message_text(message).split())
        if not text or _GREETING.match(text):
            continue

        topic = resolve_target([message])
        scanned += 1

        if topic:
            topics.append(topic)

        if len(topics) >= 4 or scanned >= 4:
            # The state line describes the thread, it does not transcribe it: four messages
            # and four topics is the whole budget.
            break

    if not topics:
        return ""

    seen: list[str] = []
    for topic in topics:
        if topic not in seen:
            seen.append(topic)
        if len(seen) >= 4:
            # Four distinct things is as much as a state line can carry, and stopping here
            # keeps the scan short on a long thread.
            break

    line = "[Earlier: " + " | ".join(seen) + "]"
    return line[:MAX_DIGEST_CHARS]


def condense(
    current_prompt: str,
    raw_history: list[object] | None = None,
    linguistics_metadata: object | None = None,
    intent_metadata: object | None = None,
) -> CondensedContextResult:
    """Decide what history this turn gets, and compress what it gets."""
    started = time.perf_counter()
    prompt = (current_prompt or "").strip()
    history = list(raw_history or [])

    raw_tokens = sum(_estimate_tokens(_message_text(message)) for message in history)

    # Nothing to condense, or nothing to condense it against.
    if not history:
        return CondensedContextResult(
            requires_history=False,
            condensed_messages=[],
            dependency_type=INDEPENDENT,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    # A greeting is not a follow-up, however much history is behind it.
    if _GREETING.match(prompt):
        return CondensedContextResult(
            requires_history=False,
            condensed_messages=[],
            dependency_type=INDEPENDENT,
            tokens_saved_estimate=raw_tokens,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    anaphora = bool(_ANAPHORA.search(prompt))
    followup = bool(_FOLLOWUP.search(prompt))

    # A deterministic, self-contained request carries its own subject. Agent 05's verdict is
    # the strongest evidence for that, so it is used before guessing from the wording.
    intent_fast = bool(getattr(intent_metadata, "is_fast_path", False))
    intent_resource = str(getattr(intent_metadata, "resource", "") or "")

    # A question about the conversation is never standalone, whatever its shape.
    history_question = bool(_HISTORY_QUESTION.search(prompt))

    if not history_question and not anaphora and not followup:
        # Two things make a turn provably standalone: a deterministic router will answer it
        # (agent 05's verdict), or it is a question that carries its own subject. Anything
        # else keeps its context - a chat product that guesses wrong here gives an answer to
        # a question the user did not ask, which is worse than the tokens it saved.
        if intent_fast or _STANDALONE_QUESTION.match(prompt):
            return CondensedContextResult(
                requires_history=False,
                condensed_messages=[],
                dependency_type=INDEPENDENT,
                tokens_saved_estimate=raw_tokens,
                telemetry_micros=(time.perf_counter() - started) * 1_000_000,
            )

    target = resolve_target(history)
    dependency = ANAPHORA_RESOLVED if anaphora else SEQUENTIAL_FOLLOWUP

    # The turn rides on what came before, so the recent turns go whole.
    recent = history[-FULL_TURNS * 2 :] if len(history) > FULL_TURNS else history
    older = history[: len(history) - len(recent)]

    condensed: list[dict[str, str]] = []

    digest = _digest_line(older) if older else ""
    if digest:
        condensed.append({"role": "system", "content": digest})

    for message in recent:
        condensed.append({"role": _message_role(message), "content": _message_text(message)})

    condensed_tokens = sum(_estimate_tokens(item["content"]) for item in condensed)
    saved = max(0, raw_tokens - condensed_tokens)

    return CondensedContextResult(
        requires_history=True,
        condensed_messages=condensed,
        resolved_target=target,
        dependency_type=dependency,
        tokens_saved_estimate=saved,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
