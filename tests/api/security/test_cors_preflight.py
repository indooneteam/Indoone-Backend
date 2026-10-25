"""A CORS preflight must not be refused by the authentication middleware.

The auth middleware runs before the CORS middleware, so if it rejects OPTIONS
the browser never receives an allow-origin header - and then never sends the
real request. That is what stopped the web app from reaching this backend.
"""
import pytest
from fastapi.testclient import TestClient

WEB_ORIGIN = "https://indooneteam.github.io"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.delenv("INDOONE_ALLOWED_ORIGINS", raising=False)

    from app.main import app

    return TestClient(app)


def test_preflight_from_the_web_origin_is_allowed(client) -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": WEB_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers.get("access-control-allow-origin") == WEB_ORIGIN


def test_preflight_from_an_unknown_origin_gets_no_allow_header(client) -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://not-ours.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") is None


def test_a_real_call_still_needs_a_token(client) -> None:
    """The preflight is open; the call itself is not."""
    response = client.post(
        "/api/chat",
        headers={"Origin": WEB_ORIGIN},
        json={"message": "hello"},
    )

    assert response.status_code == 401
