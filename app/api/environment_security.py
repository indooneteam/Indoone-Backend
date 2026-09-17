from __future__ import annotations

import os
from urllib.parse import urlparse

from cryptography.fernet import Fernet

_PRODUCTION_ENVIRONMENTS = {"prod", "production"}
_SECRET_ENV_NAMES = (
    "INDOONE_AUTH_SECRET",
    "INDOONE_APPROVAL_SECRET",
    "INDOONE_RESEARCH_TOKEN",
    "INDOONE_GOOGLE_CLIENT_SECRET",
    "INDOONE_OAUTH_ENCRYPTION_KEY",
    "INDOONE_TELEGRAM_BOT_TOKEN",
    "INDOONE_TELEGRAM_WEBHOOK_SECRET",
    "INDOONE_WHATSAPP_ACCESS_TOKEN",
    "INDOONE_WHATSAPP_VERIFY_TOKEN",
    "INDOONE_WHATSAPP_APP_SECRET",
    "INDOONE_YOUTUBE_API_KEY",
    "INDOONE_INSTAGRAM_APP_SECRET",
    "INDOONE_FACEBOOK_APP_SECRET",
    "INDOONE_CANVA_CLIENT_SECRET",
)
_PLACEHOLDER_VALUES = {
    "change-me",
    "changeme",
    "example",
    "placeholder",
    "replace-me",
    "secret",
    "test",
    "test-secret",
    "test-token",
    "your-secret",
    "your-token",
}


def _is_production() -> bool:
    return os.getenv("INDOONE_ENV", "development").strip().lower() in _PRODUCTION_ENVIRONMENTS


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDER_VALUES or normalized.startswith(("your-", "replace-", "change-"))


def _configured(name: str) -> str:
    return os.getenv(name, "").strip()


def validate_environment_security_config() -> None:
    """Fail closed on unsafe production environment and secret configuration."""
    if not _is_production():
        return

    connector_auth_required = _configured("INDOONE_CONNECTOR_AUTH_REQUIRED").lower() == "true"
    if not connector_auth_required:
        raise RuntimeError("INDOONE_CONNECTOR_AUTH_REQUIRED must be true in production")

    for name in _SECRET_ENV_NAMES:
        value = _configured(name)
        if value and _looks_like_placeholder(value):
            raise RuntimeError(f"{name} must not use a placeholder value in production")

    oauth_key = _configured("INDOONE_OAUTH_ENCRYPTION_KEY")
    if oauth_key:
        try:
            Fernet(oauth_key.encode("utf-8"))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("INDOONE_OAUTH_ENCRYPTION_KEY must be a valid Fernet key") from exc

    research_url = _configured("INDOONE_RESEARCH_URL")
    research_token = _configured("INDOONE_RESEARCH_TOKEN")
    if research_token and not research_url:
        raise RuntimeError("INDOONE_RESEARCH_URL must be configured when INDOONE_RESEARCH_TOKEN is set")
    if research_url:
        parsed = urlparse(research_url)
        if parsed.username or parsed.password:
            raise RuntimeError("INDOONE_RESEARCH_URL must not contain embedded credentials")
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError("INDOONE_RESEARCH_URL must be an HTTPS URL in production")
