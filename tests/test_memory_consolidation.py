import pytest

from app.ai.memory import extract_memory_candidates, rank_memory_matches, resolve_memory_update
from app.ai.memory_analyzer import extract_memory
from app.ai.memory_policy import MemoryPolicy, normalize_memory_key, normalize_memory_value, should_store


def test_canonical_extractor_and_legacy_adapter_match():
    canonical = extract_memory_candidates("My name is Ravi and I prefer dark mode")
    legacy = extract_memory("My name is Ravi and I prefer dark mode")
    assert [(item.key, item.value) for item in canonical] == [("name", "Ravi and I prefer dark mode")]
    assert [(item.key, item.value) for item in legacy] == [("name", "Ravi and I prefer dark mode")]


def test_policy_controls_automatic_writes():
    disabled = MemoryPolicy(allow_automatic_write=False)
    enabled = MemoryPolicy(allow_automatic_write=True, min_confidence=0.8)
    assert should_store(0.95, disabled) is False
    assert should_store(0.80, enabled) is True
    assert should_store(0.79, enabled) is False


def test_resolve_update_normalizes_and_caps_items():
    policy = MemoryPolicy(allow_automatic_write=True, min_confidence=0.8, max_items=2)
    candidates = extract_memory_candidates("My name is Ravi. I prefer dark mode.")
    updated = resolve_memory_update({"old": "value"}, candidates, policy=policy)
    assert len(updated) == 2
    assert "name" in updated


def test_memory_retrieval_uses_semantic_token_overlap_not_substring_only():
    memories = [
        {"key": "profession", "value": "software engineer", "confidence": 0.95, "updated_at": "2"},
        {"key": "hobby", "value": "photography", "confidence": 0.99, "updated_at": "1"},
    ]
    hits = rank_memory_matches(memories, "engineer profession", limit=2)
    assert hits[0]["key"] == "profession"
    assert hits[0]["match_score"] > 0


def test_memory_policy_rejects_invalid_values():
    policy = MemoryPolicy(max_value_chars=5)
    assert normalize_memory_key("  Favourite Color ") == "favourite color"
    assert normalize_memory_value("hello", policy) == "hello"
    with pytest.raises(ValueError):
        normalize_memory_value("too long", policy)
