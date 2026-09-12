from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    importance: float
    source: str = "conversation"


_PATTERNS = (
    (re.compile(r"\bmy name is\s+([^.!?\n]+)", re.I), "profile_name", 0.95),
    (re.compile(r"\bcall me\s+([^.!?\n]+)", re.I), "nickname", 0.9),
    (re.compile(r"\bi prefer\s+([^.!?\n]+)", re.I), "preference", 0.8),
    (re.compile(r"\bi like\s+([^.!?\n]+)", re.I), "interest", 0.7),
)


def extract_memory(message: str) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    text = message.strip()
    for pattern, key, importance in _PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                candidates.append(MemoryCandidate(key, value, importance))
    return candidates
