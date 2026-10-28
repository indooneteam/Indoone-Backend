"""One deterministic reading of a message, shared by every zero-token router.

The routers each grew their own regexes, and each one missed a different phrasing: "1 ralli"
reached the GitHub tool only after its own pattern was extended, "branches in 1" needed
another, and a Kanglish verb like "thorisu" meant nothing to the email path until it was
listed there too. Every gap had the same shape - a noun in one language with a postposition
from another - and the same cost: the message fell through to the model and spent tokens to
choose a tool the words had already named.

This module reads the message once, in one place:

* strips the case markers and postpositions that Kannada and Kanglish attach to a noun or
  an ordinal ("1 alli", "1 ralli", "1 ge", "branch galu"),
* maps the common Kanglish verbs onto the intent they mean ("thorisu", "helu", "kodu",
  "check maadu" -> read; "maadu", "create", "kalisi" -> write),
* extracts the ordinal, the sub-resource and the action,
* and marks the messages a fast path must leave alone.

It is pure and deterministic: no model, no network, no tokens. Callers ask it questions;
it never decides on its own to answer anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Postpositions that follow a noun or an ordinal in Kannada and Kanglish. Stripped from the
# end of a token, so "1 ralli", "1 ralli" and "1 ge" all reduce to "1".
_POSTPOSITIONS = (
    "ralli",
    "nalli",
    "alli",
    "inda",
    "ge",
    "ige",
    "annu",
    "galannu",
    "galu",
    "agalu",
    "ನಲ್ಲಿ",
    "ರಲ್ಲಿ",
    "ಅಲ್ಲಿ",
    "ಗೆ",
    "ಇಂದ",
    "ಗಳನ್ನು",
    "ಗಳು",
    "ಅನ್ನು",
)

# Verbs that mean "show me", in the three ways people actually type them.
_READ_VERBS = re.compile(
    r"\b(list|show|display|view|read|check|fetch|get|see|tell|give)\b|"
    r"\b(thorisu|torsu|helu|kodu|heli|tori|check\s*maadu|nodu)\b|"
    r"ತೋರಿಸು|ಹೇಳು|ಕೊಡು|ನೋಡು|ಚೆಕ್\s*ಮಾಡು",
    re.IGNORECASE,
)

# Verbs that change something. A fast path may route these to a confirmation, never to a
# silent read.
_WRITE_VERBS = re.compile(
    r"\b(create|make|add|open|delete|remove|merge|send|reply|forward|draft|compose|"
    r"rename|close|push|commit)\b|"
    r"\b(maadu|maadi|madi|madbeku|kalisi|kalisu|alisu|create\s*maadu|merge\s*maadu)\b|"
    r"ಮಾಡು|ಮಾಡಬೇಕು|ಕಳಿಸು|ಕಳುಹಿಸು|ಅಳಿಸು|ವಿಲೀನ",
    re.IGNORECASE,
)

_SUB_RESOURCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(pull\s*requests?|prs?)\b|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್|ಪಿಆರ್", re.IGNORECASE), "pull_requests"),
    (re.compile(r"\b(repos?|repositories|repository)\b|ರಿಪೊಸಿಟರಿ|ರೆಪೊಸಿಟರಿ", re.IGNORECASE), "repositories"),
    (re.compile(r"\b(commits?)\b|ಕಮಿಟ್", re.IGNORECASE), "commits"),
    (re.compile(r"\b(branch(?:es)?)\b|ಶಾಖೆ|ಬ್ರಾಂಚ್", re.IGNORECASE), "branches"),
    (re.compile(r"\b(issues?)\b|ಇಶ್ಯೂ", re.IGNORECASE), "issues"),
    (re.compile(r"\b(files?)\b|ಫೈಲ್", re.IGNORECASE), "files"),
)

# The mailbox, and asking about the conversation itself.
_EMAIL_WORD = re.compile(
    r"\b(mail|mails|email|emails|inbox|mailbox|unread)\b|ಮೇಲ್|ಇನ್‌ಬಾಕ್ಸ್|ಇಮೇಲ್",
    re.IGNORECASE,
)
_HISTORY_WORD = re.compile(
    r"\b(conversation|chat\s*history|history|discuss(?:ed)?|earlier)\b|"
    r"\b(maatu|matu)\b|ಸಂಭಾಷಣೆ|ಹಿಂದಿನ\s*ಮಾತು|ಚಾಟ್\s*ಹಿಸ್ಟರಿ",
    re.IGNORECASE,
)

# What a fast path must never answer: work that needs the model's judgement.
_LEAVE_ALONE = re.compile(
    r"\b(write|compose|draft|translate|summari[sz]e|explain|why|debug|refactor|"
    r"review|improve|rewrite|poem|story|essay|joke|opinion|suggest)\b|"
    r"ಬರೆ|ವಿವರಿಸು|ಅನುವಾದ|ಸಾರಾಂಶ|ಏಕೆ",
    re.IGNORECASE,
)

_ORDINAL = re.compile(r"(?<![\d#])([1-9]\d?)(?![\d%])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedQuery:
    """What the words say, read once. Empty strings and None mean "not present"."""

    text: str
    ordinal: int | None = None
    sub_resource: str = ""
    action: str = ""
    is_read: bool = False
    is_write: bool = False
    is_email: bool = False
    is_history: bool = False
    leave_alone: bool = False

    @property
    def is_github(self) -> bool:
        """A GitHub request names a repository sub-resource.

        This read ``bool(x) and x != "repositories" or x == "repositories"``, which is just
        ``bool(x)`` written the long way - the two halves cancelled out. Simplified rather
        than left to puzzle the next reader.
        """
        return bool(self.sub_resource)


_STANDALONE_POSTPOSITIONS = frozenset(
    {"ralli", "nalli", "alli", "ge", "ige", "inda", "annu", "galu", "galannu", "agalu"}
)


def _strip_suffix(token: str) -> str:
    """Normalise one token, without mangling English words.

    Two shapes exist. An attached postposition ("1ralli", "2ge") only ever appears on a
    token that starts with a digit, so stripping is limited to those - an earlier version
    stripped every token and turned "merge" into "mer" and "diverge" into "diver", which
    silently disabled the write verbs. A standalone postposition ("1 ralli", "branch galu")
    is its own token, and dropping it is enough.
    """
    lowered = token.casefold()

    if lowered in _STANDALONE_POSTPOSITIONS:
        return ""

    if not token or not token[0].isdigit():
        return token

    for suffix in sorted(_POSTPOSITIONS, key=len, reverse=True):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return token[: len(token) - len(suffix)]

    return token


def normalize(message: str) -> NormalizedQuery:
    """Read a message once: ordinal, sub-resource, action, and what to leave alone."""
    raw = (message or "").strip()

    if not raw:
        return NormalizedQuery(text="")

    # Whitespace first, then per-token suffix stripping: "1   ralli" and "1 ralli" must read
    # the same, and a suffix never spans a space.
    collapsed = _WHITESPACE.sub(" ", raw)
    tokens = [cleaned for token in collapsed.split(" ") if (cleaned := _strip_suffix(token))]
    text = " ".join(tokens)

    ordinal_match = _ORDINAL.search(text)
    ordinal = int(ordinal_match.group(1)) if ordinal_match else None

    sub_resource = ""

    for pattern, name in _SUB_RESOURCE_PATTERNS:
        if pattern.search(text):
            sub_resource = name
            break

    is_read = bool(_READ_VERBS.search(text))
    is_write = bool(_WRITE_VERBS.search(text))

    if is_read and is_write:
        # "show me how to create" is a question, not a command: the read wins, because
        # acting on a write needs an explicit verb next to the noun.
        is_write = False

    action = "write" if is_write else ("read" if is_read else "")

    return NormalizedQuery(
        text=text,
        ordinal=ordinal,
        sub_resource=sub_resource,
        action=action,
        is_read=is_read,
        is_write=is_write,
        is_email=bool(_EMAIL_WORD.search(text)),
        is_history=bool(_HISTORY_WORD.search(text)),
        leave_alone=bool(_LEAVE_ALONE.search(text)),
    )
