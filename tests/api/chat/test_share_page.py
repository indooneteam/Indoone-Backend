"""The page a share link opens.

Server-rendered, read-only, and safe with untrusted text: a shared transcript contains
whatever the model and the user wrote, so every message is escaped before anything else
happens to it.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.ai.memory.conversation_store import ConversationStore
from app.main import app


@pytest.fixture()
def shared() -> str:
    store = ConversationStore()
    conversation_id = f"page-test-{time.time_ns()}"

    store.append(
        conversation_id,
        [
            ("user", "show me a python snippet"),
            ("assistant", "Here it is:\n```python\nprint('hi')\n```\nand **bold** text."),
        ],
        user_id="user-one",
    )
    store.rename(conversation_id, "Branch cleanup", user_id="user-one")

    return store.create_share(conversation_id, user_id="user-one")


def test_the_page_renders_the_thread(shared: str) -> None:
    response = TestClient(app).get(f"/share/{shared}")

    assert response.status_code == 200
    body = response.text

    # The title the owner gave it, not a wall of bubbles with nothing to identify it.
    assert "Branch cleanup" in body
    # Both sides of the conversation.
    assert "show me a python snippet" in body
    assert "Indoone AI" in body
    # The banner and its button.
    assert "Shared conversation • Read only" in body
    assert "Try Indoone" in body
    # Branding sits above the thread title.
    assert '<div class="brand">Indoone AI</div>' in body


def test_code_and_bold_are_rendered_cleanly(shared: str) -> None:
    body = TestClient(app).get(f"/share/{shared}").text

    assert "<pre><code>" in body
    assert "print(&#x27;hi&#x27;)" in body or "print('hi')" in body
    assert "<strong>bold</strong>" in body


def test_the_page_is_read_only(shared: str) -> None:
    body = TestClient(app).get(f"/share/{shared}").text.lower()

    for control in ("<input", "<textarea", "<form", "<button"):
        assert control not in body, f"a shared page must not offer {control}"


def test_a_message_cannot_inject_markup() -> None:
    store = ConversationStore()
    conversation_id = f"page-xss-{time.time_ns()}"

    store.append(
        conversation_id,
        [("user", "<script>alert(1)</script> and <img src=x onerror=alert(2)>")],
        user_id="user-one",
    )
    share_id = store.create_share(conversation_id, user_id="user-one")

    body = TestClient(app).get(f"/share/{share_id}").text

    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body
    assert "<img src=x" not in body


def test_an_unknown_link_is_a_clean_404() -> None:
    response = TestClient(app).get("/share/" + "0" * 32)

    assert response.status_code == 404
    assert "Chat not found or link has expired" in response.text
    # Still a page, not a JSON error.
    assert "<!doctype html>" in response.text.lower()


def test_the_json_endpoint_carries_the_title(shared: str) -> None:
    payload = TestClient(app).get(f"/api/chat/shared/{shared}").json()

    assert payload["title"] == "Branch cleanup"
    assert [item["role"] for item in payload["messages"]] == ["user", "assistant"]
