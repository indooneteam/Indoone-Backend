from __future__ import annotations

from dataclasses import dataclass

from app.ai.memory import extract_memory_candidates


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    importance: float
    source: str = "conversation"


def extract_memory(message: str) -> list[MemoryCandidate]:
    """Compatibility adapter for the legacy analyzer API."""
    return [
        MemoryCandidate(
            key=candidate.key,
            value=candidate.value,
            importance=candidate.confidence,
            source=candidate.source,
        )
        for candidate in extract_memory_candidates(message)
    ]
