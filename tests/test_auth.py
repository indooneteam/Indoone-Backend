import base64
import hashlib
import hmac
import time

import pytest

from app.api.auth import extract_principal


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
