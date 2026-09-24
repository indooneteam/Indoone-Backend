from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

_DEFAULT_TOKEN_AGE_SECONDS = 3600
_DEFAULT_CLOCK_SKEW_SECONDS = 30
_PRODUCTION_ENVIRONMENTS = {"prod", "production"}


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _max_token_age_seconds() -> int:
    return _bounded_int("INDOONE_AUTH_TOKEN_MAX_AGE", _DEFAULT_TOKEN_AGE_SECONDS, minimum=60, maximum=86_400)


def _clock_skew_seconds() -> int:
    return _bounded_int("INDOONE_AUTH_CLOCK_SKEW", _DEFAULT_CLOCK_SKEW_SECONDS, minimum=0, maximum=300)


def _secret() -> bytes:
    value = os.getenv("INDOONE_AUTH_SECRET", "").strip()
    if len(value) < 32:
        raise RuntimeError("INDOONE_AUTH_SECRET must contain at least 32 characters")
    return value.encode("utf-8")


def validate_production_security_config() -> None:
    """Fail closed when production is configured without required auth secrets."""
    environment = os.getenv("INDOONE_ENV", "development").strip().lower()
    if environment not in _PRODUCTION_ENVIRONMENTS:
        return

    auth_required = os.getenv("INDOONE_AUTH_REQUIRED", "false").strip().lower() == "true"
    if not auth_required:
        raise RuntimeError("INDOONE_AUTH_REQUIRED must be true in production")

    auth_secret = _secret()
    approval_secret = os.getenv("INDOONE_APPROVAL_SECRET", "").strip()
    if len(approval_secret) < 32:
        raise RuntimeError("INDOONE_APPROVAL_SECRET must contain at least 32 characters in production")
    if auth_secret == approval_secret.encode("utf-8"):
        raise RuntimeError("INDOONE_APPROVAL_SECRET must differ from INDOONE_AUTH_SECRET")


def _decode_part(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _validate_principal(user_id: str) -> str:
    normalized = user_id.strip()
    if not normalized or len(normalized) > 256:
        raise ValueError("invalid principal")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("invalid principal")
    return normalized


_FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "indoone").strip() or "indoone"
_FIREBASE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
_FIREBASE_CERT_CACHE: dict[str, Any] = {}
_FIREBASE_CERT_CACHE_EXPIRES_AT = 0.0


def _decode_json_part(value: str) -> dict[str, Any]:
    try:
        decoded = _decode_part(value).decode("utf-8")
        payload = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid firebase token") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid firebase token")
    return payload


def _firebase_certificates() -> dict[str, Any]:
    global _FIREBASE_CERT_CACHE, _FIREBASE_CERT_CACHE_EXPIRES_AT
    now = time.time()
    if _FIREBASE_CERT_CACHE and now < _FIREBASE_CERT_CACHE_EXPIRES_AT:
        return _FIREBASE_CERT_CACHE

    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            response = client.get(_FIREBASE_CERTS_URL)
            response.raise_for_status()
            certificates = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ValueError("firebase token verification service unavailable") from exc

    if not isinstance(certificates, dict) or not certificates:
        raise ValueError("firebase token verification service returned no certificates")

    cache_control = response.headers.get("cache-control", "")
    max_age = 3600
    for directive in cache_control.split(","):
        name, separator, value = directive.strip().partition("=")
        if name.lower() == "max-age" and separator:
            try:
                max_age = max(60, int(value))
            except ValueError:
                pass
            break

    parsed: dict[str, Any] = {}
    for key_id, pem in certificates.items():
        if not isinstance(key_id, str) or not isinstance(pem, str):
            continue
        try:
            parsed[key_id] = x509.load_pem_x509_certificate(pem.encode("utf-8")).public_key()
        except ValueError:
            continue

    if not parsed:
        raise ValueError("firebase token verification service returned invalid certificates")

    _FIREBASE_CERT_CACHE = parsed
    _FIREBASE_CERT_CACHE_EXPIRES_AT = now + max_age
    return parsed


def _verify_firebase_token(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a firebase token")

    encoded_header, encoded_payload, encoded_signature = parts
    header = _decode_json_part(encoded_header)
    payload = _decode_json_part(encoded_payload)

    if header.get("alg") != "RS256":
        raise ValueError("invalid firebase token")
    key_id = header.get("kid")
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("invalid firebase token")

    public_key = _firebase_certificates().get(key_id)
    if public_key is None:
        _FIREBASE_CERT_CACHE.clear()
        raise ValueError("unknown firebase token key")

    try:
        signature = _decode_part(encoded_signature)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid firebase token") from exc

    signed_data = f"{encoded_header}.{encoded_payload}".encode("ascii")
    try:
        public_key.verify(
            signature,
            signed_data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except Exception as exc:
        raise ValueError("invalid firebase token signature") from exc

    now = int(time.time())
    issuer = f"https://securetoken.google.com/{_FIREBASE_PROJECT_ID}"
    if payload.get("aud") != _FIREBASE_PROJECT_ID or payload.get("iss") != issuer:
        raise ValueError("invalid firebase token audience or issuer")

    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise ValueError("invalid firebase token subject")
    subject = _validate_principal(subject)

    exp = payload.get("exp")
    issued_at = payload.get("iat")
    auth_time = payload.get("auth_time")
    if not isinstance(exp, (int, float)) or now >= int(exp):
        raise ValueError("expired firebase token")
    if not isinstance(issued_at, (int, float)) or int(issued_at) > now + _clock_skew_seconds():
        raise ValueError("invalid firebase token issued-at")
    if auth_time is not None and (
        not isinstance(auth_time, (int, float)) or int(auth_time) > now + _clock_skew_seconds()
    ):
        raise ValueError("invalid firebase token auth-time")

    return subject

def extract_principal(authorization: str) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ValueError("bearer authentication required")
    token = token.strip()

    try:
        return _verify_firebase_token(token)
    except ValueError:
        pass

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid bearer token")
    encoded_user, encoded_timestamp, encoded_signature = parts
    try:
        user_id = _decode_part(encoded_user).decode("utf-8")
        issued_at = int(_decode_part(encoded_timestamp).decode("ascii"))
        signature = _decode_part(encoded_signature)
    except (ValueError, UnicodeDecodeError, base64.binascii.Error) as exc:
        raise ValueError("invalid bearer token") from exc
    user_id = _validate_principal(user_id)
    age = int(time.time()) - issued_at
    if age < -_clock_skew_seconds() or age > _max_token_age_seconds():
        raise ValueError("expired bearer token")
    payload = f"{encoded_user}.{encoded_timestamp}".encode("ascii")
    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid bearer signature")
    return user_id
