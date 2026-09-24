from app.ai.conversation_store import ConversationStore


def _close_conversation(store: ConversationStore, conversation_id: str) -> None:
    messages = [
        ("user" if index % 2 == 0 else "assistant", f"message-{index}")
        for index in range(50)
    ]
    store.append(conversation_id, messages, user_id="user-1")


def test_purge_expired_closed_conversation(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    _close_conversation(store, "expired")
    with store._connect() as connection:
        connection.execute(
            "UPDATE conversations SET closed_at=datetime('now', '-8 days') WHERE conversation_id=?",
            ("expired",),
        )

    assert store.purge_expired_closed(7) == 1

    with store._connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM conversations WHERE conversation_id=?",
            ("expired",),
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM messages WHERE conversation_id=?",
            ("expired",),
        ).fetchone() is None


def test_purge_keeps_recent_closed_conversation(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    _close_conversation(store, "recent")

    assert store.purge_expired_closed(7) == 0
    with store._connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM conversations WHERE conversation_id=?",
            ("recent",),
        ).fetchone() is not None
