from pathlib import Path

import pytest

from app.ai.conversation_store import ConversationStore


def test_conversation_history_is_persistent_and_ordered(tmp_path: Path) -> None:
    db = tmp_path / "conversations.sqlite3"
    first = ConversationStore(db)
    first.append("c1", [("user", "Hello"), ("assistant", "Namaskara")])

    second = ConversationStore(db)
    assert second.recent("c1") == [("user", "Hello"), ("assistant", "Namaskara")]


def test_recent_history_is_bounded(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_messages=2)
    store.append(
        "c1",
        [("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four")],
    )

    assert store.recent("c1") == [("user", "three"), ("assistant", "four")]


def test_conversation_closes_at_message_limit(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_messages=4)
    store.append("c1", [("user", "one"), ("assistant", "two"), ("user", "three")])
    assert store.is_closed("c1") is False

    store.append("c1", [("assistant", "four")])
    assert store.message_count("c1") == 4
    assert store.is_closed("c1") is True

    with pytest.raises(ValueError, match="closed"):
        store.append("c1", [("user", "five")])


def test_invalid_messages_are_rejected(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")

    with pytest.raises(ValueError, match="role"):
        store.append("c1", [("system", "bad")])

    with pytest.raises(ValueError, match="content"):
        store.append("c1", [("user", "   ")])
