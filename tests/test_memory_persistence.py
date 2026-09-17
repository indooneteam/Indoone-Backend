from app.ai.memory_service import MemoryService
from app.ai.memory_policy import MemoryPolicy
from app.capabilities import store


def test_memory_persistence_and_user_isolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "memory.db"))

    service = MemoryService(policy=MemoryPolicy(allow_automatic_write=False))
    first = service.write_candidate(
        "user-a",
        key="name",
        value="Ravi",
        confidence=1.0,
        explicit_user_request=True,
    )
    second = service.write_candidate(
        "user-b",
        key="name",
        value="Kiran",
        confidence=1.0,
        explicit_user_request=True,
    )

    assert first is not None
    assert second is not None
    assert service.search("user-a", "name")[0]["value"] == "Ravi"
    assert service.search("user-b", "name")[0]["value"] == "Kiran"

    assert store.delete_memory("user-a", str(first["id"])) is True
    assert service.search("user-a", "name") == []
    assert service.search("user-b", "name")[0]["value"] == "Kiran"
