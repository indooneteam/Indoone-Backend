from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    confidence: float
    reason: str


_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\bmy name is\s+([^.!?]+)", "name", 0.99),
    (r"\bcall me\s+([^.!?]+)", "nickname", 0.96),
    (r"\bmy nickname is\s+([^.!?]+)", "nickname", 0.98),
    (r"\bi prefer\s+([^.!?]+)", "preference", 0.88),
    (r"\bi like\s+([^.!?]+)", "interest", 0.80),
    (r"\bmy profession is\s+([^.!?]+)", "profession", 0.98),
)


def extract_memory_candidates(message: str) -> list[MemoryCandidate]:
    text = " ".join(message.strip().split())
    if not text:
        return []
    candidates: list[MemoryCandidate] = []
    lowered = text.casefold()
    for pattern, key, confidence in _PATTERNS:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        value = text[match.start(1) : match.end(1)].strip(" \t\n.,;:!?\"'")
        if value:
            candidates.append(
                MemoryCandidate(
                    key=key,
                    value=value,
                    confidence=confidence,
                    reason="explicit_user_statement",
                )
            )
    return candidates


def resolve_memory_update(
    existing: dict[str, str],
    candidates: list[MemoryCandidate],
) -> dict[str, str]:
    updated = dict(existing)
    for candidate in candidates:
        if candidate.confidence >= 0.85:
            updated[candidate.key] = candidate.value
    return updated
