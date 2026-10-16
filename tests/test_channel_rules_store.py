from app.capabilities import store


def test_channel_rules_are_stored_per_user_and_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    created = store.upsert_channel_rule(
        user_id="user-a",
        channel="instagram",
        trigger="comment",
        keyword="Price",
        response="Here is the price list.",
    )

    assert created["channel"] == "instagram"
    assert created["trigger"] == "comment"
    assert created["keyword"] == "price"
    assert created["response"] == "Here is the price list."
    assert created["enabled"] is True
    assert created["rule_id"]

    store.upsert_channel_rule(
        user_id="user-b",
        channel="instagram",
        trigger="message",
        keyword="hello",
        response="Hi there.",
    )

    assert len(store.list_channel_rules("user-a")) == 1
    assert len(store.list_channel_rules("user-b")) == 1
    assert store.list_channel_rules("user-a", channel="whatsapp") == []


def test_channel_rule_update_keeps_identity_and_created_at(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    created = store.upsert_channel_rule(
        user_id="user-a",
        channel="instagram",
        trigger="comment",
        keyword="price",
        response="First answer.",
    )
    updated = store.upsert_channel_rule(
        user_id="user-a",
        channel="instagram",
        trigger="comment",
        keyword="price list",
        response="Second answer.",
        rule_id=str(created["rule_id"]),
    )

    assert updated["rule_id"] == created["rule_id"]
    assert updated["created_at"] == created["created_at"]
    assert updated["keyword"] == "price list"
    assert updated["response"] == "Second answer."
    assert len(store.list_channel_rules("user-a")) == 1


def test_channel_rule_disable_and_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    created = store.upsert_channel_rule(
        user_id="user-a",
        channel="instagram",
        trigger="message",
        keyword="hours",
        response="We are open 9 to 6.",
    )
    rule_id = str(created["rule_id"])

    assert store.set_channel_rule_enabled("user-a", rule_id, False) is True
    stored = store.get_channel_rule("user-a", rule_id)
    assert stored is not None
    assert stored["enabled"] is False
    assert store.list_channel_rules("user-a", enabled_only=True) == []

    assert store.delete_channel_rule("user-a", rule_id) is True
    assert store.get_channel_rule("user-a", rule_id) is None
    assert store.delete_channel_rule("user-a", rule_id) is False


def test_channel_rule_validation_rejects_bad_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    rejected = (
        {"user_id": "", "channel": "instagram", "trigger": "comment", "keyword": "a", "response": "b"},
        {"user_id": "user-a", "channel": "", "trigger": "comment", "keyword": "a", "response": "b"},
        {"user_id": "user-a", "channel": "instagram", "trigger": "story", "keyword": "a", "response": "b"},
        {"user_id": "user-a", "channel": "instagram", "trigger": "comment", "keyword": "a", "response": "   "},
    )

    for payload in rejected:
        try:
            store.upsert_channel_rule(**payload)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {payload}")


def test_channel_rule_limit_is_enforced_per_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))

    for index in range(store._CHANNEL_RULE_MAX_PER_CHANNEL):
        store.upsert_channel_rule(
            user_id="user-a",
            channel="instagram",
            trigger="comment",
            keyword=f"keyword-{index}",
            response=f"answer-{index}",
        )

    try:
        store.upsert_channel_rule(
            user_id="user-a",
            channel="instagram",
            trigger="comment",
            keyword="one-too-many",
            response="not stored",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected the per-channel rule limit to be enforced")

    # The limit is per channel, so another channel still accepts a rule.
    store.upsert_channel_rule(
        user_id="user-a",
        channel="whatsapp",
        trigger="message",
        keyword="hello",
        response="Hi there.",
    )
    assert len(store.list_channel_rules("user-a", channel="whatsapp")) == 1
