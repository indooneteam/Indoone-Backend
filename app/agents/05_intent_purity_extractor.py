"""Agent 05 - the Intent Purity Extractor: what is being asked, and can it be answered free.

Two jobs, both deterministic:

* **Strip the grammar.** "1 ralli" means branch one; "repo 2 ge" means repository two;
  "branches galannu thorisu" means show the branches. The case marker is glue, not meaning,
  and anything keying on the noun has to see the noun.
* **Decide whether a model is needed at all.** A holiday question, a branch list and an
  unread-mail count are answered by a router that costs nothing. Saying so here, with a
  confidence, is what lets the pipeline skip a provider call instead of paying for a
  decision the words already made.

The stripper is deliberately timid, and that is the important part. An earlier version of
this idea stripped a suffix whenever a word ended in one, which turned "merge" into "mer"
and "diverge" into "diver". So a Romanised suffix is removed only when what remains is a
resource this agent knows, or when the token began with a digit. Kannada script suffixes are
removed freely because ಅಲ್ಲಿ and ಗೆ are never the end of an English verb.

This agent does not route anything by itself. The existing routers already answer these
questions, and adding a second router that also answers them is how a message ends up
claimed twice. It reports its verdict so the pipeline can log the case that matters: a
message this agent is sure is answerable for free, that no router actually answered.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

FAST_PATH_FLOOR = 0.92

# ---------------------------------------------------------------- vocabulary

# The nouns the stripper is allowed to rebuild. Nothing outside this set is rewritten, which
# is what keeps "merge", "diverge" and "message" intact.
_RESOURCES: dict[str, str] = {
    "branch": "branch", "branches": "branch",
    "repo": "repository", "repos": "repository", "repository": "repository",
    "repositories": "repository",
    "pr": "pull_request", "prs": "pull_request", "pullrequest": "pull_request",
    "pullrequests": "pull_request",
    "commit": "commit", "commits": "commit",
    "issue": "issue", "issues": "issue",
    "diff": "diff", "diffs": "diff",
    "calendar": "holiday_calendar", "holiday": "holiday_calendar",
    "holidays": "holiday_calendar", "festival": "holiday_calendar",
    "festivals": "holiday_calendar",
    "mail": "email", "mails": "email", "email": "email", "emails": "email",
    "inbox": "email",
}

# Romanised case markers. Only stripped behind a known resource or a digit.
_ROMAN_MARKERS = (
    "galannu", "annu", "vannu", "nannu", "galu", "gala", "dalli", "ralli", "nalli", "alli",
    "kke", "ige", "ge", "mein", "ko", "se", "ka", "ke", "ki", "ku", "la", "ai", "lo", "nu",
)

# Kannada script markers, longest first so "ಗಳನ್ನು" is not eaten by "ಗಳು".
_KANNADA_MARKERS = ("ಗಳನ್ನು", "ಗಳಿಗೆ", "ಗಳಲ್ಲಿ", "ಗಳು", "ಗಳ", "ಅನ್ನು", "ಇಗೆ", "ಕ್ಕೆ", "ದಲ್ಲಿ", "ನಲ್ಲಿ", "ಅಲ್ಲಿ", "ಗೆ")

_ACTIONS: dict[str, tuple[str, ...]] = {
    "LIST": ("list", "show", "get", "display", "thorisu", "torusu", "heli", "helu", "dikhao",
             "batao", "kaattu", "chupinchu", "tೋrisu"),
    "DIFF": ("diff", "difference", "compare", "badla", "change", "enu change", "kya badla"),
    "CHECK_HOLIDAY": ("holiday", "festival", "raja", "chutti", "pandaga", "habba", "calendar",
                      "deepavali", "ugadi", "dasara", "holidays"),
    "READ_EMAIL": ("unread", "inbox", "mail", "mails", "email", "emails", "patra"),
}

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "ondaneya": 1, "eradaneya": 2, "mooraneya": 3, "nalkaneya": 4,
    "pehla": 1, "pehli": 1, "doosra": 2, "doosri": 2, "teesra": 3,
    "ondu": 1, "eradu": 2, "mooru": 3,
}

_TARGETS = {
    "holiday_calendar": "calendar",
    "branch": "github_read",
    "repository": "github_read",
    "pull_request": "github_read",
    "commit": "github_read",
    "diff": "github_read",
    "email": "email_read",
}

_WORD = re.compile(r"[a-z]+|\d+")

# Words that carry no intent of their own. Their presence does not make a request impure.
_STOPWORDS = frozenset(
    {"when", "is", "the", "a", "an", "me", "my", "please", "and", "of", "in", "on", "for",
     "to", "what", "which", "are", "was", "it", "this", "that", "do", "does", "i", "you",
     "how", "many", "much", "any", "all", "new", "latest", "current"}
)


@dataclass(slots=True, frozen=True)
class Telemetry:
    execution_microseconds: int


@dataclass(slots=True, frozen=True)
class IntentVerdict:
    """What was asked, and whether it can be answered without a model."""

    is_fast_path: bool
    fast_path_target: str | None
    action: str
    resource: str
    extracted_ordinal: int | None
    extracted_identifier: str | None
    confidence_score: float
    cleaned_query: str
    telemetry: Telemetry


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_INTENT_EXTRACTOR_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------- the stripper


def _strip_roman(token: str) -> str:
    """Remove a Romanised case marker only when the remainder is a noun we know."""
    lowered = token.casefold()

    if lowered in _RESOURCES:
        # "branches" becomes "branch": the plural is grammar, and every caller wants the noun.
        return _RESOURCES[lowered].replace("repository", "repo").replace("pull_request", "pr")

    # A marker standing on its own is glue with nothing to glue: "1 ralli" -> "1".
    if lowered in _ROMAN_MARKERS:
        return ""

    for marker in _ROMAN_MARKERS:
        if not lowered.endswith(marker) or len(lowered) <= len(marker) + 1:
            continue

        stem = lowered[: -len(marker)]

        # A digit keeps its marker's meaning: "2 ge" is the second one.
        if stem.isdigit():
            return stem

        # Otherwise the remainder has to be a resource we recognise. This is the rule that
        # stops "merge" becoming "mer".
        if stem in _RESOURCES:
            return _RESOURCES[stem].replace("repository", "repo").replace("pull_request", "pr")

    return token


def _strip_kannada(token: str) -> str:
    """Kannada case markers are never the tail of an English verb, so this is free."""
    for marker in _KANNADA_MARKERS:
        if token.endswith(marker) and len(token) > len(marker) + 1:
            return token[: -len(marker)]
    return token


def clean(query: str) -> str:
    """The query with its grammar removed and its nouns intact."""
    parts: list[str] = []

    for token in query.split():
        core = token.strip(".,!?;:\"'()[]")
        if not core:
            continue
        if any("\u0c80" <= character <= "\u0cff" for character in core):
            parts.append(_strip_kannada(core))
        elif core.isascii():
            stripped = _strip_roman(core)
            if stripped:
                parts.append(stripped)
        else:
            parts.append(core)

    return " ".join(parts)


# ---------------------------------------------------------------- the reading


def _ordinal(query: str) -> int | None:
    """A number only when the whole token is a number.

    Splitting on letters and digits turns a commit hash like 632a7f6d09 into "632", "7",
    "6", "09" and reports branch seven, so the token is judged whole.
    """
    for token in query.split():
        if token.isdigit() and 0 < int(token) < 100:
            return int(token)

    for token in query.split():
        if token.casefold() in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[token.casefold()]

    return None


def _action(lowered: str, words: set[str]) -> tuple[str, bool]:
    if any(marker in words or marker in lowered for marker in _ACTIONS["CHECK_HOLIDAY"]):
        return "CHECK_HOLIDAY", True

    if any(marker in words for marker in _ACTIONS["DIFF"]):
        return "DIFF", True

    email_named = any(marker in words for marker in _ACTIONS["READ_EMAIL"])
    reading = any(marker in words for marker in _ACTIONS["LIST"])
    if email_named and reading:
        return "READ_EMAIL", True
    if email_named:
        return "READ_EMAIL", True
    if reading:
        return "LIST", True

    return "GENERAL_CHAT", False


def extract(message: str) -> IntentVerdict:
    """Read one message. No model, no network, no file, no thread."""
    started = time.perf_counter()
    query = message or ""
    cleaned = clean(query)
    lowered = cleaned.casefold()
    words = set(_WORD.findall(lowered))

    action, action_found = _action(lowered, words)

    resource = "NONE"
    if action == "CHECK_HOLIDAY":
        # A holiday question names the calendar whatever else it says: "when is deepavali"
        # carries no word from the resource list and is still the calendar's question.
        resource = "HOLIDAY_CALENDAR"
    else:
        for word in _WORD.findall(lowered):
            if word in _RESOURCES:
                resource = _RESOURCES[word].upper()
                break

    ordinal = _ordinal(cleaned)
    identifier = None
    quoted = re.search(r"[\"'`]([^\"'`]{2,40})[\"'`]", query)
    if quoted:
        identifier = quoted.group(1).strip()

    # Confidence is built from what was actually found, never assumed. A question with a
    # recognised action AND a recognised resource is the only kind that can reach the floor.
    # Purity: a request made only of a recognised action and a recognised resource is the
    # one case that can be answered without reading it again. Every word accounted for is
    # the evidence; one unknown content word is enough to send it to the model.
    recognised = set(_RESOURCES) | set(_ROMAN_MARKERS) | _STOPWORDS | set(_ORDINAL_WORDS)
    for markers in _ACTIONS.values():
        recognised |= set(markers)

    content = [word for word in _WORD.findall(lowered) if word not in _STOPWORDS]
    pure = bool(content) and all(word in recognised or word.isdigit() for word in content)

    confidence = 0.0
    if action_found:
        confidence += 0.35
    if resource != "NONE":
        confidence += 0.45
    if ordinal is not None or identifier:
        confidence += 0.10
    if pure:
        confidence += 0.15
    if len(words) <= 8:
        confidence += 0.05
    if action == "GENERAL_CHAT" or resource == "NONE":
        confidence = min(confidence, 0.55)

    confidence = round(min(1.0, confidence), 3)

    target = _TARGETS.get(resource.casefold()) if resource != "NONE" else None
    is_fast = bool(target) and confidence >= FAST_PATH_FLOOR

    return IntentVerdict(
        is_fast_path=is_fast,
        fast_path_target=target if is_fast else None,
        action=action,
        resource=resource,
        extracted_ordinal=ordinal,
        extracted_identifier=identifier,
        confidence_score=confidence,
        cleaned_query=cleaned,
        telemetry=Telemetry(
            execution_microseconds=int((time.perf_counter() - started) * 1_000_000),
        ),
    )
