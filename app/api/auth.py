from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

_DEFAULT_TOKEN_AGE_SECONDS = 3600
_DEFAULT_CLOCK_SKEW_SECONDS = 30


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


def _decode_part(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def extract_principal(authorization: str) -> str:
    if not authorization.startswith("Bearer "):
        raise ValueError("bearer authentication required")
    token = authorization[7:].strip()
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
    if not user_id or len(user_id) > 256:
        raise ValueError("invalid principal")
    age = int(time.time()) - issued_at
    if age < -_clock_skew_seconds() or age > _max_token_age_seconds():
        raise ValueError("expired bearer token")
    payload = f"{encoded_user}.{encoded_timestamp}".encode("ascii")
    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid bearer signature")
    return user_id
