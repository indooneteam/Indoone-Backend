import base64

import pytest
from cryptography.fernet import Fernet

from app.api.environment_security import validate_environment_security_config


def _set_production(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "a" * 48)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "b" * 48)
    monkeypatch.setenv("INDOONE_ALLOWED_HOSTS", "api.example.com")


def test_production_rejects_placeholder_secret(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_YOUTUBE_API_KEY", "changeme")

    with pytest.raises(RuntimeError, match="INDOONE_YOUTUBE_API_KEY"):
        validate_environment_security_config()


def test_production_accepts_non_placeholder_optional_secret(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_YOUTUBE_API_KEY", "prod-youtube-key-123")

    validate_environment_security_config()


def test_production_validates_oauth_encryption_key(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", base64.urlsafe_b64encode(b"x" * 32).decode())

    validate_environment_security_config()


def test_production_rejects_invalid_oauth_encryption_key(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", "not-a-fernet-key")

    with pytest.raises(RuntimeError, match="valid Fernet key"):
        validate_environment_security_config()


def test_production_requires_research_url_for_research_token(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_RESEARCH_TOKEN", "prod-research-token")

    with pytest.raises(RuntimeError, match="INDOONE_RESEARCH_URL"):
        validate_environment_security_config()


def test_production_research_url_requires_https(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_RESEARCH_URL", "http://research.example.com/search")

    with pytest.raises(RuntimeError, match="HTTPS URL"):
        validate_environment_security_config()


def test_production_research_url_rejects_embedded_credentials(monkeypatch) -> None:
    _set_production(monkeypatch)
    monkeypatch.setenv("INDOONE_RESEARCH_URL", "https://user:pass@research.example.com/search")

    with pytest.raises(RuntimeError, match="embedded credentials"):
        validate_environment_security_config()


def test_env_files_are_ignored_except_example() -> None:
    lines = open(".gitignore", encoding="utf-8").read().splitlines()

    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines


def test_fernet_key_generation_matches_documented_format() -> None:
    key = Fernet.generate_key().decode("ascii")
    assert len(key) == 44
