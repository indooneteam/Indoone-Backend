from app.ai.memory_policy import MemoryPolicy
from app.ai.memory_service import MemoryService


def test_memory_service_blocks_implicit_write(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "memory.db"))
    service = MemoryService()

    assert service.write_candidate(
        "user-a", key="name", value="Ravi", confidence=0.99
    ) is None


def test_memory_service_allows_explicit_write_and_replaces_same_key(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "memory.db"))
    service = MemoryService()

    first = service.write_candidate(
        "user-a", key="name", value="Ravi", confidence=0.99, explicit_user_request=True
    )
    second = service.write_candidate(
        "user-a", key="name", value="Rahul", confidence=0.99, explicit_user_request=True
    )

    assert first is not None
    assert second is not None
    assert second["id"] == first["id"]
    assert second["value"] == "Rahul"
    assert len(service.search("user-a", "name", limit=10)) == 1


def test_memory_service_isolates_users_and_supports_forget(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "memory.db"))
    service = MemoryService()

    alice = service.write_candidate(
        "alice", key="profession", value="software engineer", confidence=0.95, explicit_user_request=True
    )
    bob = service.write_candidate(
        "bob", key="profession", value="teacher", confidence=0.95, explicit_user_request=True
    )

    assert alice is not None and bob is not None
    assert service.search("alice", "engineer", limit=10)[0]["value"] == "software engineer"
    assert service.search("bob", "engineer", limit=10) == []
    assert service.delete("alice", str(alice["id"])) is True
    assert service.search("alice", "engineer", limit=10) == []
    assert service.delete("alice", str(alice["id"])) is False
    assert service.search("bob", "profession", limit=10)[0]["value"] == "teacher"


def test_memory_service_enforces_item_limit_but_allows_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "memory.db"))
    service = MemoryService(MemoryPolicy(max_items=2))

    assert service.write_candidate(
        "user-a", key="name", value="Ravi", confidence=0.99, explicit_user_request=True
    ) is not None
    assert service.write_candidate(
        "user-a", key="profession", value="engineer", confidence=0.99, explicit_user_request=True
    ) is not None
    assert service.write_candidate(
        "user-a", key="hobby", value="photography", confidence=0.99, explicit_user_request=True
    ) is None
    assert service.write_candidate(
        "user-a", key="name", value="Rahul", confidence=0.99, explicit_user_request=True
    ) is not None
    assert len(service.search("user-a", "", limit=10)) == 2
