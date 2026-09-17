import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.network_security import configure_network_security, validate_network_security_config


def test_development_uses_safe_default_hosts(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.delenv("INDOONE_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("INDOONE_TRUST_PROXY_HEADERS", "false")

    app = FastAPI()
    app.get("/")(lambda: {"status": "ok"})
    configure_network_security(app)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/", headers={"host": "evil.example"}).status_code == 400


def test_production_requires_explicit_host_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.delenv("INDOONE_ALLOWED_HOSTS", raising=False)

    with pytest.raises(RuntimeError, match="INDOONE_ALLOWED_HOSTS must be configured"):
        validate_network_security_config()


def test_production_rejects_wildcard_network_trust(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_ALLOWED_HOSTS", "api.example.com")
    monkeypatch.setenv("INDOONE_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("INDOONE_TRUST_PROXY_HEADERS", "true")
    monkeypatch.setenv("INDOONE_TRUSTED_PROXY_IPS", "*")

    with pytest.raises(RuntimeError, match="wildcard"):
        validate_network_security_config()


def test_production_accepts_explicit_network_trust(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_ALLOWED_HOSTS", "api.example.com,api.internal.example")
    monkeypatch.setenv("INDOONE_ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("INDOONE_TRUST_PROXY_HEADERS", "true")
    monkeypatch.setenv("INDOONE_TRUSTED_PROXY_IPS", "127.0.0.1")

    validate_network_security_config()


def test_cors_allows_only_configured_origin(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "development")
    monkeypatch.setenv("INDOONE_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("INDOONE_ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("INDOONE_TRUST_PROXY_HEADERS", "false")

    app = FastAPI()
    app.get("/")(lambda: {"status": "ok"})
    configure_network_security(app)
    client = TestClient(app)

    allowed = client.options(
        "/",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    blocked = client.options(
        "/",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert blocked.status_code == 400
