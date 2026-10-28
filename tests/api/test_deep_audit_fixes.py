"""Three defects the deep audit found, held by tests.

1. A shared conversation is public by design, but the auth gate answered every share link
   with 401 as soon as INDOONE_AUTH_REQUIRED was on - which production runs.
2. The capability endpoints trusted a caller-supplied `user_id`, so any authenticated
   caller could read or write another person's projects, tasks and agent runs.
3. The Google News RSS provider caught `ElementTree.ParseError`, and `ElementTree` was
   never imported - a malformed feed raised NameError instead of the intended RuntimeError.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.content.capabilities import _scoped_user
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


class _State:
    def __init__(self, principal: str = "") -> None:
        self.principal_id = principal


class _Request:
    def __init__(self, principal: str = "") -> None:
        self.state = _State(principal)


def test_a_supplied_user_id_must_match_the_token() -> None:
    with pytest.raises(HTTPException) as refused:
        _scoped_user(_Request("user-one"), "user-two")

    assert refused.value.status_code == 403


def test_a_matching_user_id_passes() -> None:
    assert _scoped_user(_Request("user-one"), "user-one") == "user-one"


def test_without_a_principal_the_supplied_id_is_used() -> None:
    # A local checkout runs with INDOONE_AUTH_REQUIRED unset; behaviour there is unchanged.
    assert _scoped_user(_Request(""), "user-one") == "user-one"


def test_the_share_page_is_public_even_when_auth_is_required(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")

    # No Authorization header at all: the link has to work for someone without an account.
    response = TestClient(app).get("/share/" + "0" * 32)

    # 404 is the page saying "not found"; 401 would be the gate refusing the reader.
    assert response.status_code == 404
    assert "Chat not found" in response.text


def test_the_api_still_requires_a_token_when_auth_is_required(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")

    assert TestClient(app).get("/api/conversations").status_code == 401


def test_a_malformed_feed_is_a_runtime_error_not_a_name_error() -> None:
    from app.web_research import research

    provider = research.GoogleNewsRssResearchProvider()

    class _Response:
        content = b"<rss><channel><item>"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            return _Response()

    def _client(*_args, **_kwargs):
        return _Client()

    import asyncio

    original = research.httpx.AsyncClient
    research.httpx.AsyncClient = _client

    try:
        with pytest.raises(RuntimeError) as failed:
            asyncio.run(provider.search("karnataka", limit=3))
    finally:
        research.httpx.AsyncClient = original

    assert "RSS" in str(failed.value)
