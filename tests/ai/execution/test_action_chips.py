"""A held write offers the two answers it understands, as data.

The reply already said "Reply CONFIRM to go ahead, or CANCEL to stop" - so the user had
to type a word to press a button that did not exist. The chips are that button: the same
word, carried as a field, so pressing and typing cannot diverge.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.execution import action_chips, tools
from app.plugins.github.chat_tools import _format_github_result


@pytest.fixture(autouse=True)
def _collecting(monkeypatch):
    # A hold issues a signed token; without the secret it takes the APPROVAL_UNAVAILABLE
    # branch instead, which is a different case entirely.
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "s" * 32)
    action_chips.begin()
    yield
    action_chips.reset()


def test_a_held_write_offers_confirm_and_cancel(monkeypatch) -> None:
    # The hold signs a token and stores the pending action. Neither is what this test is
    # about, and a missing database would take the APPROVAL_UNAVAILABLE branch instead -
    # which correctly offers no buttons, so the assertion would never see a chip.
    monkeypatch.setattr(
        "app.capabilities.connectors.store.save_pending_github_action",
        lambda *_args, **_kwargs: None,
    )

    class _FakeFernet:
        def encrypt(self, value: bytes) -> bytes:
            return value

        def decrypt(self, value: bytes) -> bytes:
            return value

    monkeypatch.setattr(
        "app.capabilities.connectors.integrations._fernet", lambda: _FakeFernet()
    )

    held = tools._hold_write(
        {
            "repository": "owner/repo",
            "branch": "feat/x",
            "conversation_id": "c1",
            "original_message": "what does this repo look like",
        },
        "u1",
        "github_delete_branch",
    )

    assert held is not None
    assert json.loads(held)["code"] == "APPROVAL_REQUIRED"

    chips = action_chips.collected()
    assert [chip["label"] for chip in chips] == ["Confirm", "Cancel"]
    # The action is the exact text the endpoint resolves, not a private code.
    assert [chip["action"] for chip in chips] == ["CONFIRM", "CANCEL"]


def test_a_branch_list_offers_the_one_follow_up_that_needs_no_guessing() -> None:
    _format_github_result(
        "github_list_branches",
        {
            "ok": True,
            "result": {
                "repository": "owner/repo",
                "branches": [{"name": "main", "sha": "a" * 40, "protected": False}],
            },
        },
    )

    chips = action_chips.collected()
    # Three follow-ups, each carrying a complete request rather than a name to retype.
    assert [chip["label"] for chip in chips] == [
        "Create a branch",
        "View commits",
        "Create a pull request",
    ]
    for chip in chips:
        assert "owner/repo" in chip["action"]


def test_an_ordinary_reply_carries_nothing() -> None:
    _format_github_result(
        "github_get_commit",
        {"ok": True, "result": {"sha": "a" * 40, "message": "hello"}},
    )

    assert action_chips.collected() == []


def test_collection_is_off_unless_a_request_asked_for_it() -> None:
    action_chips.reset()

    # A worker thread or a test calling an executor directly must not accumulate chips
    # for whatever request happens to run next.
    action_chips.emit(action_chips.APPROVE)

    assert action_chips.collected() == []


def test_the_chat_endpoint_returns_them() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    import base64
    import hashlib
    import hmac
    import os
    import time

    os.environ["INDOONE_AUTH_SECRET"] = "x" * 32

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    user = encode(b"user-one")
    stamp = encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{stamp}".encode("ascii")
    signature = encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    headers = {"Authorization": f"Bearer {user}.{stamp}.{signature}"}

    from app.api.chat import chat as chat_module

    async def fake_reply(*_args, **_kwargs):
        return "a plain answer"

    import types

    # Stubbed reply: this environment has no trained local model, and the assertion is
    # about the response shape, not about the answer.
    monkeypatch_target = types.SimpleNamespace(generate_reply=fake_reply)
    _ = monkeypatch_target
    chat_module.generate_reply = fake_reply

    response = client.post(
        "/api/chat",
        json={"message": "delete the branch feat/x in owner/repo"},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert "action_chips" in body
    assert isinstance(body["action_chips"], list)
    assert body["action_chips"] == []
