"""The stream carries the provider's chunks, not a finished string chopped up.

The old emitter sliced the complete reply into 48-character pieces and paused between
them - a replay that looked like streaming. These tests hold the two things that matter:
the deltas are exactly what the generator produced, and the completion hook still carries
the chips and the final text.

Provider-level streaming itself cannot be verified here: it is a live HTTP contract with
Gemini and Azure. What is verified is the pipeline that would carry it.
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


def test_the_deltas_are_the_generator_chunks_not_forty_eight_character_slices(monkeypatch) -> None:
    async def fake_router(*_args, **_kwargs):
        return None

    async def fake_loop(*_args, **_kwargs):
        return None

    async def fake_stream(*_args, **_kwargs):
        for chunk in ("Hel", "lo the", "re, friend"):
            yield chunk

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)
    monkeypatch.setattr("app.api.chat.platform.run_tool_loop", fake_loop)
    monkeypatch.setattr("app.api.chat.platform.stream_generate", fake_stream)

    response = client.post(
        "/api/chat/stream",
        json={"message": "say hello"},
        headers=_auth_headers(),
    )

    events = _events(response.text)
    deltas = [event["text"] for event in events if event.get("type") == "delta"]

    # Exactly the chunks the generator produced - not 48-character windows of the joined
    # string, which is what the replay produced.
    assert deltas == ["Hel", "lo the", "re, friend"]
    assert events[-1]["reply"] == "Hello there, friend"


def test_a_tool_answer_is_one_chunk_and_keeps_its_chips(monkeypatch) -> None:
    async def fake_router(*_args, **_kwargs):
        action_chips.emit(action_chips.APPROVE, action_chips.CANCEL)
        return "I will delete the branch feat/x. Reply CONFIRM to go ahead, or CANCEL to stop."

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)

    response = client.post(
        "/api/chat/stream",
        json={"message": "delete the branch feat/x in owner/repo"},
        headers=_auth_headers(),
    )

    events = _events(response.text)
    deltas = [event["text"] for event in events if event.get("type") == "delta"]

    # A deterministic answer has nothing to reveal gradually.
    assert len(deltas) == 1
    assert [chip["label"] for chip in events[-1]["action_chips"]] == ["Confirm", "Cancel"]


def test_a_provider_failure_mid_answer_ends_the_stream_with_a_message(monkeypatch) -> None:
    async def fake_router(*_args, **_kwargs):
        return None

    async def fake_loop(*_args, **_kwargs):
        return None

    async def failing_stream(*_args, **_kwargs):
        yield "partial"
        raise RuntimeError("provider dropped the connection")

    monkeypatch.setattr("app.api.chat.platform.handle_github_chat_with_tools", fake_router)
    monkeypatch.setattr("app.api.chat.platform.run_tool_loop", fake_loop)
    monkeypatch.setattr("app.api.chat.platform.stream_generate", failing_stream)

    response = client.post(
        "/api/chat/stream",
        json={"message": "say hello"},
        headers=_auth_headers(),
    )

    events = _events(response.text)

    # Headers are already sent, so a 503 is impossible. The last event carries the failure
    # and the client replaces what it streamed with it - half an answer must not stand.
    assert response.status_code == 200
    assert events[-1]["type"] == "done"
    assert "could not" in events[-1]["reply"].lower() or events[-1]["reply"]
    assert events[-1]["action_chips"] == []


def test_the_gateway_falls_back_to_one_chunk_without_provider_streaming(monkeypatch) -> None:
    import asyncio

    from app.ai.gateway import service

    async def fake_reply(*_args, **_kwargs):
        return "a complete answer"

    monkeypatch.delenv("INDOONE_STREAM_PROVIDERS", raising=False)
    monkeypatch.setattr(service, "generate_reply", fake_reply)

    async def collect():
        return [chunk async for chunk in service.stream_generate("hi")]

    assert asyncio.run(collect()) == ["a complete answer"]
