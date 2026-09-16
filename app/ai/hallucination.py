"""Deterministic checks for unsupported claims of external actions or access."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class HallucinationQuality:
    """Result of the narrow external-action hallucination check."""

    passed: bool
    reason: str = ""
    matches: tuple[str, ...] = ()


_ACTION_VERBS = r"(?:checked|accessed|verified|searched|opened|called|sent|looked at)"
_ACTION_PATTERNS = (
    rf"\bi\s+{_ACTION_VERBS}\b",
    rf"\bi\s+have\s+{_ACTION_VERBS}\b",
    rf"\band\s+{_ACTION_VERBS}\b",
)
_ACTION_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _ACTION_PATTERNS), re.IGNORECASE)
_NEGATION_RE = re.compile(
    rf"\b(?:i\s+(?:didn'?t|did not|can'?t|cannot|can not|haven'?t|have not))\s+{_ACTION_VERBS}\b",
    re.IGNORECASE,
)


def assess_hallucination(
    answer: str,
    allowed_external_actions: list[str] | tuple[str, ...] = (),
) -> HallucinationQuality:
    """Reject affirmative external-action claims not backed by an allow-list."""

    cleaned = answer.strip()
    if not cleaned:
        return HallucinationQuality(True)

    allowed = {item.casefold().strip() for item in allowed_external_actions if item.strip()}
    matches = [match.group(0) for match in _ACTION_RE.finditer(cleaned)]
    if not matches:
        return HallucinationQuality(True)

    effective_matches = [match for match in matches if not _NEGATION_RE.search(match)]
    if not effective_matches:
        return HallucinationQuality(True)

    if not allowed:
        return HallucinationQuality(False, "unsupported_external_action_claim", tuple(effective_matches))

    unsupported = [match for match in effective_matches if match.casefold() not in allowed]
    if unsupported:
        return HallucinationQuality(False, "unsupported_external_action_claim", tuple(unsupported))

    return HallucinationQuality(True, matches=tuple(effective_matches))
