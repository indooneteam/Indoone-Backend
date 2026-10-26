"""A GitHub question is answered by the router, not by the model's memory.

The report was precise: a branch list came back in a fraction of a second, with names
that were never in the repository. That is not a slow API - it is no API call at all.

The deterministic router already existed; it only ran on the local model backend. On
Gemini and Azure - production - a GitHub request went to the model loop, where calling a
tool is the model's choice, and a model that answers from memory invents branch names
instantly. These tests hold the router in front of the model on every backend.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = _encode(user_id.encode("utf-8"))
    timestamp = _encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)


def test_a_github_request_is_routed_before_the_model(monkeypatch) -> None:
    """The router answers; the model is never asked about repository state."""

    seen = {}

    async def fake_router(user_id, message, **kwargs):
        seen["message"] = message
        return "indoone has 4 branches: main, develop, terminal, web"

    async def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("the model must not be asked for repository state")

    monkeypatch.setattr("app.api.chat.chat.handle_github_chat_with_tools", fake_router)
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setattr("app.api.chat.chat.run_tool_loop", model_must_not_run)

    response = client.post(
        "/api/chat",
        json={"message": "branch list kodu"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["reply"].startswith("indoone has 4 branches")
    assert seen["message"] == "branch list kodu"


def test_an_ordinary_message_still_reaches_the_model(monkeypatch) -> None:
    """The router returns None for everything else, so the model path is untouched."""

    async def no_github_request(*_args, **_kwargs):
        return None

    called = {}

    async def fake_model(*_args, **kwargs):
        called["ran"] = True
        return "a plain answer"

    monkeypatch.setattr("app.api.chat.chat.handle_github_chat_with_tools", no_github_request)
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setattr("app.api.chat.chat.run_tool_loop", fake_model)

    response = client.post(
        "/api/chat",
        json={"message": "explain what a pull request is"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert called.get("ran") is True
