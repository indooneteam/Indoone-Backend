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

    for marker in (" — ", " - ", " · ", "\t"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()

    # A trailing "(main)" on a repository line is the default branch, not part of the name.
    if value.endswith(")") and " (" in value:
        value = value.split(" (", 1)[0].strip()

    return value.strip()


def parse_numbered_items(text: str) -> list[str]:
    """Every numbered item in a message, in order."""
    items: list[str] = []

    for line in (text or "").splitlines():
        match = _NUMBERED.match(line)

        if not match:
            continue

        cleaned = _clean_item(match.group(2))

        if cleaned:
            items.append(cleaned)

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


def resolve_from_history(value: str, history: list[tuple[str, str]] | None) -> str | None:
    """Resolve a value against the most recent assistant message that listed something.

    Only the previous answers are read, never the user's own words - the user's message is
    what is being resolved, so reading it back would resolve it to itself.
    """
    if not history:
        return None

    for role, text in reversed(history):
        if role != "assistant":
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
