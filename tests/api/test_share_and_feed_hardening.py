"""Three holes found by a deep audit: an unbounded public page, and two in the feed reader.

The share page is public, unauthenticated and was unlimited - a long thread cost the
server whatever it cost. The feed reader parsed untrusted XML with the stdlib parser and
opened whatever scheme a source named.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai.memory.conversation_store import SHARED_MESSAGE_LIMIT, ConversationStore
from app.main import app


def _long_thread(count: int) -> str:
    from uuid import uuid4

    store = ConversationStore()
    # A unique thread per call: the store keeps what earlier runs wrote, and a fixed id
    # made the second run of this file see twice the messages.
    conversation = f"share-cap-{uuid4().hex}"
    store.append(conversation, [("user", "first")], user_id="owner-cap")
    store.append(
        conversation,
        [(("user" if index % 2 else "assistant"), f"message {index}") for index in range(count)],
        user_id="owner-cap",
    )
    store.rename(conversation, "A long thread", user_id="owner-cap")
    return store.create_share(conversation, user_id="owner-cap")


def test_the_public_page_does_not_render_an_unbounded_thread() -> None:
    share_id = _long_thread(SHARED_MESSAGE_LIMIT + 10)

    body = TestClient(app).get(f"/share/{share_id}").text

    assert body.count('class="bubble') <= SHARED_MESSAGE_LIMIT
    assert f"Showing the last {SHARED_MESSAGE_LIMIT} messages" in body


def test_the_json_view_is_capped_too() -> None:
    share_id = _long_thread(SHARED_MESSAGE_LIMIT + 10)

    payload = TestClient(app).get(f"/api/chat/shared/{share_id}").json()

    assert len(payload["messages"]) <= SHARED_MESSAGE_LIMIT


def test_a_short_thread_is_not_cut() -> None:
    share_id = _long_thread(5)

    body = TestClient(app).get(f"/share/{share_id}").text

    assert "Showing the last" not in body
    assert body.count('class="bubble') == 6


def test_an_entity_expansion_feed_is_refused() -> None:
    from app.ai.local import self_update

    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>'
        b"<rss><channel><item><title>&lol2;</title></item></channel></rss>"
    )
    source = self_update.SourceDefinition(
        name="test", url="https://example.com/feed", allowed_hosts=["example.com"]
    )

    with pytest.raises(ValueError):
        self_update.parse_feed(bomb, source)


def test_a_non_http_source_is_refused() -> None:
    from app.ai.local import self_update

    source = self_update.SourceDefinition(
        name="local", url="file:///etc/passwd", allowed_hosts=["localhost"]
    )

    with pytest.raises(ValueError):
        self_update.fetch_feed(source)
