from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.ai.memory import extract_memory_candidates, rank_memory_matches, resolve_memory_update
from app.ai.memory_policy import MemoryPolicy, normalize_memory_key, normalize_memory_value, should_store
from app.capabilities import store


class MemoryService:
    def __init__(self, policy: MemoryPolicy = MemoryPolicy()) -> None:
        self.policy = policy

    def extract(self, message: str):
        return extract_memory_candidates(message)

    def propose_update(self, existing: Mapping[str, str], message: str) -> dict[str, str]:
        return resolve_memory_update(existing, self.extract(message), policy=self.policy)

    def write_candidate(
        self,
        user_id: str,
        *,
        key: str,
        value: str,
        confidence: float,
        source: str = "conversation",
        explicit_user_request: bool = False,
    ) -> dict[str, Any] | None:
        if not self.policy.allow_automatic_write and not explicit_user_request:
            return None
        if not 0.0 <= confidence <= 1.0 or confidence < self.policy.min_confidence:
            return None
        if self.policy.allow_automatic_write and not should_store(confidence, self.policy):
            return None
        normalized_key = normalize_memory_key(key)
        normalized_value = normalize_memory_value(value, self.policy)
        current = store.list_memories(user_id, limit=self.policy.max_items + 1)
        existing = {str(item["key"]): str(item["value"]) for item in current if "key" in item and "value" in item}
        merged = resolve_memory_update(
            existing,
            extract_memory_candidates(f"my {normalized_key} is {normalized_value}"),
            policy=self.policy,
        )
        if normalized_key not in existing and len(existing) >= self.policy.max_items:
            return None
        if len(merged) > self.policy.max_items:
            return None
        return store.upsert_memory(user_id, normalized_key, normalized_value, confidence, source)

    def search(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, object]]:
        limit = max(1, min(limit, self.policy.max_items))
        memories = store.list_memories(user_id, limit=self.policy.max_items)
        return rank_memory_matches(memories, query, limit=limit)

    def delete(self, user_id: str, memory_id: str) -> bool:
        return store.delete_memory(user_id, memory_id)
