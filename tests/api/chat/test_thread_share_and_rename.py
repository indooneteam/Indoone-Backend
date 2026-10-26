"""Sharing a thread, renaming it, and reading a shared one.

The share link reads the conversation live rather than copying it, so the tests check the
two things that follow from that: the link shows what is actually in the thread, and it
cannot be used to write.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

import pytest
from fastapi.testclient import TestClient

from app.ai.memory.conversation_store import ConversationStore
from app.main import app


def auth_headers(user: bytes = b"user-one") -> dict[str, str]:
    os.environ["INDOONE_AUTH_SECRET"] = "x" * 32

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    subject = encode(user)
    stamp = encode(str(int(time.time())).encode("ascii"))
    payload = f"{subject}.{stamp}".encode("ascii")
    signature = encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())

    return {"Authorization": f"Bearer {subject}.{stamp}.{signature}"}


@pytest.fixture()
def thread() -> str:
    store = ConversationStore()
    conversation_id = f"share-test-{time.time_ns()}"
    store.append(
        conversation_id,
        [("user", "hello there"), ("assistant", "hello, how can I help?")],
        user_id="user-one",
    )
    return conversation_id


def test_a_share_link_is_minted_and_reads_the_thread(thread: str) -> None:
    client = TestClient(app)

    shared = client.post(f"/api/chat/threads/{thread}/share", headers=auth_headers())
    assert shared.status_code == 200

    share_url = shared.json()["share_url"]
    share_id = share_url.rsplit("/", 1)[-1]
    # Non-guessable: 32 hex characters from uuid4, not a counter.
    assert len(share_id) == 32 and all(char in "0123456789abcdef" for char in share_id)

    # Public: no Authorization header at all.
    read = client.get(f"/api/chat/shared/{share_id}")
    assert read.status_code == 200

    messages = read.json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "hello there"
    # Nothing about the owner is in the payload.
    assert "user_id" not in read.json()
    assert "conversation_id" not in read.json()


def test_an_unknown_share_id_is_a_404() -> None:
    assert TestClient(app).get("/api/chat/shared/" + "0" * 32).status_code == 404


def test_a_shared_thread_cannot_be_written_to(thread: str) -> None:
    client = TestClient(app)
    share_id = (
        client.post(f"/api/chat/threads/{thread}/share", headers=auth_headers())
        .json()["share_url"]
        .rsplit("/", 1)[-1]
    )

    # There is no write verb on the shared path, and the read path rejects one.
    assert client.post(f"/api/chat/shared/{share_id}", json={"text": "hi"}).status_code == 405
    assert client.patch(f"/api/chat/shared/{share_id}", json={"title": "x"}).status_code == 405


def test_a_thread_can_be_renamed(thread: str) -> None:
    response = TestClient(app).patch(
        f"/api/chat/threads/{thread}", json={"title": "  Branch   cleanup  "}, headers=auth_headers()
    )

    assert response.status_code == 200
    # Cleaned on the way in: one space between words, no padding.
    assert response.json()["status"] == "Branch cleanup"


@pytest.mark.parametrize("title", ["", "   ", "\n"])
def test_an_empty_title_is_rejected(thread: str, title: str) -> None:
    response = TestClient(app).patch(
        f"/api/chat/threads/{thread}", json={"title": title}, headers=auth_headers()
    )

    assert response.status_code == 400


def test_another_users_thread_is_refused(thread: str) -> None:
    other = auth_headers(b"user-two")

    assert TestClient(app).post(f"/api/chat/threads/{thread}/share", headers=other).status_code == 403
    assert (
        TestClient(app)
        .patch(f"/api/chat/threads/{thread}", json={"title": "mine"}, headers=other)
        .status_code
        == 403
    )
