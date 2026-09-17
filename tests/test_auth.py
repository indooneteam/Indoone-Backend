import base64
import hashlib
import hmac
import time

import pytest

from app.api.auth import extract_principal, validate_production_security_config


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _token(user_id: str, issued_at: int) -> str:
    user = _encode(user_id.encode("utf-8"))
    timestamp = _encode(str(issued_at).encode("ascii"))
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return f"{user}.{timestamp}.{signature}"


def test_extract_principal_accepts_bearer_scheme_case_insensitively(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    token = _token("user-one", int(time.time()))
    assert extract_principal(f"bEaReR {token}") == "user-one"


def test_extract_principal_rejects_expired_token(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    token = _token("user-one", int(time.time()) - 7200)
    with pytest.raises(ValueError, match="expired"):
        extract_principal(f"Bearer {token}")


def test_extract_principal_rejects_whitespace_in_identity(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    token = _token("user one", int(time.time()))
    with pytest.raises(ValueError, match="invalid principal"):
        extract_principal(f"Bearer {token}")


def test_production_security_config_requires_global_auth(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "a" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "b" * 32)

    with pytest.raises(RuntimeError, match="INDOONE_AUTH_REQUIRED must be true"):
        validate_production_security_config()


def test_production_security_config_requires_strong_separate_secrets(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "a" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a" * 32)

    with pytest.raises(RuntimeError, match="must differ"):
        validate_production_security_config()


def test_production_security_config_accepts_secure_configuration(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "a" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "b" * 32)

    validate_production_security_config()
