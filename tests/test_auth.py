import base64
import hashlib
import hmac
import time

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.api import auth as auth_module
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


def _firebase_token(private_key: rsa.RSAPrivateKey, user_id: str = "firebase-user", **overrides: object) -> str:
    now = int(time.time())
    header = {"alg": "RS256", "kid": "test-key", "typ": "JWT"}
    payload = {
        "iss": "https://securetoken.google.com/indoone",
        "aud": "indoone",
        "sub": user_id,
        "iat": now,
        "exp": now + 3600,
        "auth_time": now,
        **overrides,
    }
    import json

    encoded_header = _encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signed = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = private_key.sign(
        signed,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return f"{encoded_header}.{encoded_payload}.{_encode(signature)}"


def test_extract_principal_accepts_firebase_id_token(monkeypatch) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        auth_module,
        "_firebase_certificates",
        lambda: {"test-key": private_key.public_key()},
    )
    token = _firebase_token(private_key, "firebase-user")
    assert extract_principal(f"Bearer {token}") == "firebase-user"


def test_extract_principal_rejects_firebase_wrong_audience(monkeypatch) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        auth_module,
        "_firebase_certificates",
        lambda: {"test-key": private_key.public_key()},
    )
    token = _firebase_token(private_key, "firebase-user", aud="other-project")
    with pytest.raises(ValueError, match="invalid bearer"):
        extract_principal(f"Bearer {token}")
