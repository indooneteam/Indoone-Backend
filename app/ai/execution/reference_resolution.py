"""Turning "1" and "indoone" into the thing the user meant.

A list came back numbered - branches, repositories - and the next message was "2". The
router demanded `owner/name` and refused, so the user had to retype a name they had just
been shown. This module reads the numbered items out of the previous answer and resolves a
selection against them.

Deliberately narrow: it resolves a value that IS a selection ("2", "the second one",
"indoone" when exactly one item contains it). It never guesses a verb - a bare "1" with no
request around it is a question for the user, not something to act on.
"""

from __future__ import annotations

import re

_NUMBERED = re.compile(r"^\s*(\d{1,3})[.)]\s+(.+?)\s*$")
# A repository URL on its own line: the item above it is that repository.
_REPOSITORY_URL = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "last": -1,
}


def _clean_item(text: str) -> str:
    """The name from a numbered line, without the annotations after it.

    ``1. indooneteam/indoone (main) — Public``  ->  ``indooneteam/indoone``
    ``1. main · a1b2c3d · protected``           ->  ``main``
    """
    value = text.strip()

    # A repository is now shown as [name](url). The link target is the only place the
    # full owner/name survives, so the ordinal still resolves to something a tool can use.
    link = re.search(r"\[([^\]]+)\]\(([^)]+)\)", value)

    if link:
        target = re.search(r"github\.com/([^/]+)/([^/?#\s]+)", link.group(2))

        if target:
            return f"{target.group(1)}/{target.group(2)}"

        return link.group(1).strip()

    for marker in (" — ", " - ", " · ", "\t"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()

    # A trailing "(main)" on a repository line is the default branch, not part of the name.
    if value.endswith(")") and " (" in value:
        value = value.split(" (", 1)[0].strip()

    return value.strip()


def parse_numbered_items(text: str) -> list[str]:
    """Every numbered item in a message, in order.

    A repository list puts the URL on the line under the name, so the link can keep its own
    row and its copy button. When that line is a GitHub URL, the item resolves to
    owner/name - the form a tool can actually use - rather than to the bare label.
    """
    lines = (text or "").splitlines()
    items: list[str] = []

    for index, line in enumerate(lines):
        match = _NUMBERED.match(line)

        if not match:
            continue

        cleaned = _clean_item(match.group(2))
        from_url = ""

        # The first non-blank line after the item may be its URL - unless it is the next
        # numbered item, in which case the item has no URL of its own and must not borrow
        # the one below it.
        for follow in lines[index + 1 : index + 3]:
            if not follow.strip():
                continue

            if _NUMBERED.match(follow):
                break

            found = _REPOSITORY_URL.search(follow)

            if found:
                from_url = f"{found.group(1)}/{found.group(2)}"

            break

        resolved = from_url or cleaned

        if resolved:
            items.append(resolved)

    return items


def _ordinal_index(value: str) -> int | None:
    """1-based index for "2", "#2", "second", "2nd", "last"; None when it is not one."""
    text = (value or "").strip().casefold().rstrip(".")

    if not text:
        return None

    digits = text.lstrip("#").strip()

    if digits.isdigit():
        return int(digits)

    word = text.removeprefix("the ").strip()
    # "the first one", "the second one": the trailing "one" is not part of the ordinal.
    word = re.sub(r"\s+one$", "", word).strip()

    return _ORDINAL_WORDS.get(word)


def resolve_selection(value: str, items: list[str]) -> str | None:
    """The item this value selects, or None when it selects nothing or too much.

    An ordinal is taken literally. A partial name resolves only when exactly one item
    contains it - two matches is ambiguity, and picking one of them would be a guess
    dressed up as an answer.
    """
    if not items:
        return None

    index = _ordinal_index(value)

    if index is not None:
        if index == -1:
            return items[-1]
        if 1 <= index <= len(items):
            return items[index - 1]
        return None

    needle = (value or "").strip().casefold()

    if not needle:
        return None

    matches = [item for item in items if needle in item.casefold()]

    if len(matches) == 1:
        return matches[0]

    return None


def _looks_like_repository_list(text: str) -> bool:
    """A repository list carries GitHub URLs; a branch list carries names only."""
    return bool(_REPOSITORY_URL.search(text or ""))


def resolve_from_history(
    value: str,
    history: list[tuple[str, str]] | None,
    *,
    kind: str = "",
) -> str | None:
    """Resolve a value against the most recent assistant message that listed something.

    ``kind`` keeps the two lists apart. A repository ordinal must never be resolved against
    a branch list: with a branch list of "1. develop, 2. main, 3. terminal", "3" was being
    read as the third repository and the wrong repository was opened.

    Only the previous answers are read, never the user's own words - the user's message is
    what is being resolved, so reading it back would resolve it to itself.
    """
    if not history:
        return None

    for role, text in reversed(history):
        if role != "assistant":
            continue

        if kind == "repository" and not _looks_like_repository_list(text):
            continue

        if kind == "branch" and _looks_like_repository_list(text):
            continue

        items = parse_numbered_items(text)

        if not items:
            continue

        resolved = resolve_selection(value, items)

        if resolved is not None:
            return resolved

    return None


def selection_token(message: str) -> str:
    """The bare selection in a message - "2", "the second one", "last" - or "".

    A number inside a name (``v2``, ``feat/2``) is not a selection, so the token must stand
    on its own: digits alone, or an ordinal word with nothing attached.
    """
    text = (message or "").strip()

    if not text:
        return ""

    match = re.search(
        r"(?<![\w/.-])(#?\d{1,2})(?![\w/.-])|\b(the\s+)?(first|second|third|fourth|fifth|last)(\s+one)?\b",
        text,
        re.IGNORECASE,
    )

    if not match:
        return ""

    if match.group(1):
        return match.group(1)

    return match.group(3)


# A repository name as it appears anywhere in a sentence: owner/name.
_REPOSITORY_IN_TEXT = re.compile(r"\b([A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100})\b")


def recent_repository(
    history: list[tuple[str, str]] | None, turns: int = 3
) -> str:
    """The repository the conversation is already about, or "".

    "Branch estu ede adralli" names no repository, and answering it with "Which repository
    should I open?" makes the user repeat something the thread already settled. The last few
    turns are read for a repository name - either listed or named in prose - and the most
    recent one is the active target.

    Only ``turns`` messages are read, so a repository discussed ten messages ago is not
    silently reused for a question that has moved on.
    """
    if not history:
        return ""

    window = [text for _role, text in history[-max(1, turns):] if text]

    for text in reversed(window):
        # A GitHub URL is the reliable form: the generic owner/name matcher also matches
        # "github.com/other-org" inside a link, which is not a repository.
        found = _REPOSITORY_URL.findall(text)

        if found:
            # The last mention in the message is the one being discussed.
            owner, repo = found[-1]
            return f"{owner}/{repo}"

        named = _REPOSITORY_IN_TEXT.findall(text)

        if named:
            return named[-1]

    return ""
