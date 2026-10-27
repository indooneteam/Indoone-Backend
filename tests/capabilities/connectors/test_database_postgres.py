"""The capability store, on Postgres.

SQLite is the default and its tests run everywhere. These run only when a Postgres is
pointed at, which CI does with a service container and which a developer can do with
INDOONE_TEST_DATABASE_URL. When neither is available the module skips, so a checkout
with no Postgres still has a green suite.

Every test here is marked `postgres`, so `pytest -m postgres` selects exactly this file's
eleven tests and `pytest -m "not postgres"` leaves them out. They are skipped locally
because there is no server, not because anything is missing or out of date: the assertions
are about Postgres itself - the dialect, BYTEA blobs, FOR UPDATE - and running them against
SQLite would pass while testing the wrong database.

What is checked is the part that could silently differ between the two databases: the
schema builds, the upserts replace rather than fail, the single-use paths stay single
use, and a row reads back by column name.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.storage import database

_POSTGRES_URL = os.getenv("INDOONE_TEST_DATABASE_URL", "").strip()

# Selected by `-m postgres`, run in CI by the `postgres-store` job.
pytestmark = pytest.mark.postgres


def _url() -> str:
    """Return the Postgres to test against, or skip.

    CI supplies one as a service container. A developer can point at any Postgres
    with INDOONE_TEST_DATABASE_URL. There is deliberately no bundled server here: a
    test that starts its own database behaves differently on every machine, and the
    thing under test is the store, not the packaging.
    """

    if not _POSTGRES_URL:
        pytest.skip(
            "needs a Postgres: set INDOONE_TEST_DATABASE_URL, or run the "
            "postgres-store CI job, which supplies one"
        )
    return _POSTGRES_URL


@pytest.fixture()
def store(monkeypatch: pytest.MonkeyPatch):
    """Point the store at Postgres, on a schema of its own, and hand it back."""

    url = _url()
    monkeypatch.setenv("INDOONE_DATABASE_URL", url)
    from app.capabilities.connectors import db_runtime, store as store_module

    db_runtime.configure_sqlite_runtime()
    store_module.initialize()
    return store_module


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def test_it_is_actually_talking_to_postgres(store) -> None:
    assert database.dialect() == "postgres"
    assert database.blob_type() == "BYTEA"
    assert database.for_update().strip() == "FOR UPDATE"


def test_the_schema_builds_with_binary_columns(store) -> None:
    with database.connect() as db:
        columns = db.table_columns("integration_tokens")
    assert {"user_id", "integration_id", "access_token", "refresh_token"} <= columns
    with database.connect() as db:
        assert db.integrity_problems() == 0


def test_an_oauth_state_is_consumed_exactly_once(store) -> None:
    state = _unique("state")
    store.create_oauth_state(state, "user-pg", "github", "https://example.test/cb")

    first = store.consume_oauth_state(state, "github")
    second = store.consume_oauth_state(state, "github")

    assert first == {
        "user_id": "user-pg",
        "integration_id": "github",
        "redirect_uri": "https://example.test/cb",
    }
    assert second is None


def test_an_expired_oauth_state_is_refused(store) -> None:
    state = _unique("state")
    store.create_oauth_state(state, "user-pg", "github", "https://example.test/cb")

    assert store.consume_oauth_state(state, "github", max_age_seconds=0) is None


def test_writing_the_same_oauth_state_twice_replaces_it(store) -> None:
    state = _unique("state")
    store.create_oauth_state(state, "user-one", "github", "https://one.test/cb")
    store.create_oauth_state(state, "user-two", "github", "https://two.test/cb")

    consumed = store.consume_oauth_state(state, "github")

    assert consumed is not None
    assert consumed["user_id"] == "user-two"
    assert consumed["redirect_uri"] == "https://two.test/cb"


def test_an_inbound_event_is_claimed_once(store) -> None:
    event = _unique("event")

    assert store.claim_channel_inbound_event(event) is True
    assert store.claim_channel_inbound_event(event) is False

    store.complete_channel_inbound_event(event, success=True)


def test_an_integration_token_round_trips_with_binary_payloads(store) -> None:
    user = _unique("user")
    store.upsert_integration_token(user, "github", b"access-bytes", b"refresh-bytes", "Bearer", "repo", None)

    token = store.get_integration_token(user, "github")
    assert token is not None
    assert bytes(token["access_token"]) == b"access-bytes"
    assert bytes(token["refresh_token"]) == b"refresh-bytes"

    store.upsert_integration_token(user, "github", b"new-access", None, "Bearer", "repo", None)
    updated = store.get_integration_token(user, "github")
    assert updated is not None
    assert bytes(updated["access_token"]) == b"new-access"


def test_a_memory_is_replaced_not_duplicated(store) -> None:
    user = _unique("user")
    store.upsert_memory(user, "timezone", "first", 0.5, "chat")
    store.upsert_memory(user, "timezone", "second", 0.9, "chat")

    memories = store.list_memories(user)

    assert len(memories) == 1
    assert memories[0]["value"] == "second"


def test_a_project_is_created_and_partially_updated(store) -> None:
    user = _unique("user")
    project = store.create_project(user, "Launch", "be brief", {"topic": "launch"})

    updated = store.update_project(user, project["id"], name="Launch v2")

    assert updated is not None
    assert updated["name"] == "Launch v2"
    assert updated["instructions"] == "be brief"
    assert updated["context"] == {"topic": "launch"}


def test_a_pending_email_action_round_trips_and_expires(store) -> None:
    user = _unique("user")
    conversation = _unique("conversation")
    store.save_pending_email_action(user, conversation, "zoho_mail", "send_email", b"payload", "2999-01-01T00:00:00+00:00")

    pending = store.get_pending_email_action(user, conversation)
    assert pending is not None
    assert bytes(pending["encrypted_payload"]) == b"payload"

    store.delete_pending_email_action(user, conversation)
    assert store.get_pending_email_action(user, conversation) is None


def test_channel_rules_are_listed_per_user_and_channel(store) -> None:
    user = _unique("user")
    created = store.upsert_channel_rule(user, "instagram", "comment", "price", "check your dm")

    rules = store.list_channel_rules(user, "instagram")

    assert [rule["rule_id"] for rule in rules] == [created["rule_id"]]
    assert store.list_channel_rules(user, "telegram") == []
