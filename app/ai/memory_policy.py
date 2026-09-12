from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryPolicy:
    max_items: int = 10
    max_value_chars: int = 4_000
    min_confidence: float = 0.55
    allow_automatic_write: bool = False


def should_store(confidence: float, policy: MemoryPolicy = MemoryPolicy()) -> bool:
    return policy.allow_automatic_write and policy.min_confidence <= confidence <= 1.0
