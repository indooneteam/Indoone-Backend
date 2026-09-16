from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from app.ai.memory_policy import MemoryPolicy, normalize_memory_key, normalize_memory_value


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    confidence: float
    reason: str
    source: str = "conversation"


_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\bmy name is\s+([^.!?\n]+)", "name", 0.99),
    (r"\bcall me\s+([^.!?\n]+)", "nickname", 0.96),
    (r"\bmy nickname is\s+([^.!?\n]+)", "nickname", 0.98),
    (r"\bi prefer\s+([^.!?\n]+)", "preference", 0.88),
    (r"\bi like\s+([^.!?\n]+)", "interest", 0.80),
    (r"\bmy profession is\s+([^.!?\n]+)", "profession", 0.98),
)


def extract_memory_candidates(message: str) -> list[MemoryCandidate]:
    text = " ".join(message.strip().split())
    if not text:
        return []
    candidates: list[MemoryCandidate] = []
    for pattern, key, confidence in _PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip(" \t\n.,;:!?\"'")
        if value:
            candidates.append(
                MemoryCandidate(
                    key=normalize_memory_key(key),
                    value=value,
                    confidence=confidence,
                    reason="explicit_user_statement",
                )
            )
    return candidates


def resolve_memory_update(
    existing: Mapping[str, str],
    candidates: list[MemoryCandidate],
    *,
    policy: MemoryPolicy = MemoryPolicy(),
) -> dict[str, str]:
    """Resolve an in-memory candidate update; persistence policy is enforced at write time."""
    updated = {
        normalize_memory_key(key): normalize_memory_value(value, policy)
        for key, value in existing.items()
    }
    for candidate in candidates:
        normalized_key = normalize_memory_key(candidate.key)
        normalized_value = normalize_memory_value(candidate.value, policy)
        updated[normalized_key] = normalized_value
    if len(updated) > policy.max_items:
        updated = dict(list(updated.items())[-policy.max_items :])
    return updated


def rank_memory_matches(
    memories: list[Mapping[str, object]],
    query: str,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Rank already user-scoped memory records by token overlap and confidence."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    query_tokens = {token.casefold() for token in re.findall(r"[\w'-]+", query) if len(token) > 1}
    if not query_tokens:
        return []

    scored: list[tuple[float, float, str, dict[str, object]]] = []
    for raw in memories:
        key = str(raw.get("key", ""))
        value = str(raw.get("value", ""))
        source = str(raw.get("source", ""))
        tokens = {token.casefold() for token in re.findall(r"[\w'-]+", f"{key} {value} {source}") if len(token) > 1}
        overlap = len(query_tokens & tokens) / len(query_tokens)
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
        if overlap <= 0:
            continue
        score = 0.7 * overlap + 0.3 * confidence
        item = dict(raw)
        item["match_score"] = score
        scored.append((score, confidence, str(raw.get("updated_at", "")), item))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored[:limit]]
