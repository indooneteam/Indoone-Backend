from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryPolicy:
    max_items: int = 10
    max_value_chars: int = 4_000
    min_confidence: float = 0.55
    allow_automatic_write: bool = False

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError("max_items must be at least 1")
        if self.max_value_chars < 1:
            raise ValueError("max_value_chars must be at least 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


def should_store(confidence: float, policy: MemoryPolicy = MemoryPolicy()) -> bool:
    return policy.allow_automatic_write and 0.0 <= confidence <= 1.0 and confidence >= policy.min_confidence


def normalize_memory_value(value: str, policy: MemoryPolicy = MemoryPolicy()) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) > policy.max_value_chars:
        raise ValueError("memory value exceeds policy limit")
    return normalized


def normalize_memory_key(key: str) -> str:
    normalized = " ".join(str(key).split()).strip().lower()
    if not normalized:
        raise ValueError("memory key cannot be empty")
    if len(normalized) > 200:
        raise ValueError("memory key exceeds policy limit")
    return normalized
