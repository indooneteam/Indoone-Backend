"""The streaming endpoint is the same pipeline as /api/chat, not a lesser copy.

It used to be a parallel path: no admission control, no GitHub router, no chips, its own
conversation store. Switching a client to it would have traded the corrections we just
shipped - real branch data, Confirm/Cancel - for the appearance of streaming.

These tests hold the two things that matter: the router runs before the model on this
endpoint too, and the last event carries the chips the reply offered.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.ai.execution import action_chips
from app.main import app


client = TestClient(app)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = _encode(user_id.encode("utf-8"))
    stamp = _encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{stamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{stamp}.{signature}"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)


def _events(body: str) -> list[dict]:
    parsed = []

    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            parsed.append(json.loads(line[len("data: "):]))
        except ValueError:
            continue

    return parsed


def test_the_router_answers_before_the_model_and_the_chips_arrive(monkeypatch) -> None:
    async def fake_router(user_id, message, **kwargs):
        action_chips.emit(action_chips.APPROVE, action_chips.CANCEL)
        return "I will delete the branch feat/x in owner/repo. Reply CONFIRM to go ahead, or CANCEL to stop."

    async def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("the model must not be asked for repository state")

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)
    monkeypatch.setattr("app.api.chat.platform.run_tool_loop", model_must_not_run)

    response = client.post(
        "/api/chat/stream",
        json={"message": "delete the branch feat/x in owner/repo"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    events = _events(response.text)

    kinds = [event.get("type") for event in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"

    done = events[-1]
    assert [chip["label"] for chip in done["action_chips"]] == ["Confirm", "Cancel"]
    # The finished reply rides along, so a client that only streams keeps the text.
    assert "CONFIRM" in done["reply"]


def test_the_streamed_text_reassembles_to_the_reply(monkeypatch) -> None:
    async def fake_router(*_args, **_kwargs):
        return "Hello there, this is a reply long enough to need more than one chunk of text."

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)

    response = client.post(
        "/api/chat/stream",
        json={"message": "hi"},
        headers=_auth_headers(),
    )

    events = _events(response.text)
    streamed = "".join(event.get("text", "") for event in events if event.get("type") == "delta")

    assert streamed == events[-1]["reply"]


def test_a_plain_answer_carries_no_chips(monkeypatch) -> None:
    async def fake_router(*_args, **_kwargs):
        return None

    async def fake_loop(*_args, **_kwargs):
        return "a plain answer"

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)
    monkeypatch.setattr("app.api.chat.platform.run_tool_loop", fake_loop)

    response = client.post(
        "/api/chat/stream",
        json={"message": "explain what a pull request is"},
        headers=_auth_headers(),
    )

    events = _events(response.text)
    assert events[-1]["action_chips"] == []
