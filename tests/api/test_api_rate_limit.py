"""The API burst limit: a script is stopped, a person never notices.

The chat limiter already protected the expensive path. Nothing protected the cheap ones -
/api/projects, /api/tasks, /api/agent/runs, the share page - so they could be walked as
fast as a client could ask.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai.execution import limits
from app.main import app


def _clear_buckets() -> None:
    with limits._api_lock:
        limits._api_buckets.clear()


@pytest.fixture()
def limited(monkeypatch):
    monkeypatch.setenv("INDOONE_API_LIMITS_ENABLED", "true")
    monkeypatch.setenv("INDOONE_API_RATE_BURST", "3")
    monkeypatch.setenv("INDOONE_API_RATE_PER_MINUTE", "1")
    _clear_buckets()

    yield

    _clear_buckets()


def test_the_burst_is_spent_and_then_refused(limited) -> None:
    client = TestClient(app)
    codes = [client.get("/api/projects?user_id=u1").status_code for _ in range(4)]

    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_a_refusal_says_how_long_to_wait(limited) -> None:
    client = TestClient(app)

    for _ in range(3):
        client.get("/api/projects?user_id=u1")

    refused = client.get("/api/projects?user_id=u1")

    assert refused.status_code == 429
    assert int(refused.headers.get("Retry-After", "0")) >= 1


def test_the_share_page_is_not_the_api(limited) -> None:
    # The page is public and cheap; it is the API that needed the limit.
    assert TestClient(app).get("/share/" + "0" * 32).status_code == 404


def test_the_limit_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_API_LIMITS_ENABLED", "false")
    monkeypatch.setenv("INDOONE_API_RATE_BURST", "1")
    _clear_buckets()

    client = TestClient(app)

    assert [client.get("/api/projects?user_id=u1").status_code for _ in range(3)] == [200, 200, 200]


def test_the_bucket_refills_over_time(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_API_RATE_BURST", "2")
    monkeypatch.setenv("INDOONE_API_RATE_PER_MINUTE", "60")
    _clear_buckets()

    # Two tokens, then a refusal; 120 seconds later the bucket is full again.
    first = limits.take_api_burst("ip:refill", now=0.0)
    second = limits.take_api_burst("ip:refill", now=0.0)
    blocked = limits.take_api_burst("ip:refill", now=0.0)
    refilled = limits.take_api_burst("ip:refill", now=120.0)

    assert first.allowed and second.allowed
    assert not blocked.allowed
    assert refilled.allowed


def test_the_limits_can_be_read_from_outside() -> None:
    """The numbers live in environment variables; this endpoint says what they are."""
    body = TestClient(app).get("/api/ops/limits").json()

    assert set(body) >= {"chat", "api", "configured_by", "documented_in"}
    assert "per_minute" in body["chat"] and "burst" in body["chat"]
    assert "per_minute" in body["api"] and "burst" in body["api"]
    # The names are listed so the reader knows which variable to change.
    assert "INDOONE_API_RATE_PER_MINUTE" in body["configured_by"]
