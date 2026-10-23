from __future__ import annotations

import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

_PRODUCTION_ENVIRONMENTS = {"prod", "production"}
_DEFAULT_ALLOWED_HOSTS = ("testserver", "localhost", "127.0.0.1")

# The project's own web app, served from GitHub Pages. Its origin is fixed and
# known, and without it the web app cannot reach this backend at all: a browser
# refuses the call before it is sent. The native app is unaffected, because it
# does not make cross-origin calls.
#
# This one is always allowed, alongside whatever INDOONE_ALLOWED_ORIGINS names.
# It was tried as a fallback first, and that was not enough: the variable is
# already set on the running server, and a set value won, so the web app stayed
# blocked. The variable now adds to this origin rather than replacing it.
_WEB_APP_ORIGIN = "https://indooneteam.github.io"


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_production() -> bool:
    return os.getenv("INDOONE_ENV", "development").strip().lower() in _PRODUCTION_ENVIRONMENTS


def allowed_hosts() -> list[str]:
    configured = _csv_env("INDOONE_ALLOWED_HOSTS")
    if configured:
        return configured
    return [] if _is_production() else list(_DEFAULT_ALLOWED_HOSTS)


def allowed_origins() -> list[str]:
    """The project's own web origin, plus whatever the server configures.

    The web origin is always present, so the web app works whether or not
    INDOONE_ALLOWED_ORIGINS has been set. The variable adds any further
    origins a deployment needs.
    """
    origins = [_WEB_APP_ORIGIN]

    for origin in _csv_env("INDOONE_ALLOWED_ORIGINS"):
        if origin not in origins:
            origins.append(origin)

    return origins


def trusted_proxy_ips() -> list[str]:
    return _csv_env("INDOONE_TRUSTED_PROXY_IPS") or ["127.0.0.1"]


def validate_network_security_config() -> None:
    """Reject unsafe production host, CORS, and proxy-header configuration."""
    hosts = _csv_env("INDOONE_ALLOWED_HOSTS")
    origins = allowed_origins()
    trust_proxy_headers = os.getenv("INDOONE_TRUST_PROXY_HEADERS", "false").strip().lower() == "true"
    proxies = _csv_env("INDOONE_TRUSTED_PROXY_IPS")

    if "*" in hosts:
        raise RuntimeError("INDOONE_ALLOWED_HOSTS must not contain wildcard '*'")
    if _is_production():
        if not hosts:
            raise RuntimeError("INDOONE_ALLOWED_HOSTS must be configured in production")
        if "*" in origins:
            raise RuntimeError("INDOONE_ALLOWED_ORIGINS must not contain wildcard '*' in production")
        if trust_proxy_headers and not proxies:
            raise RuntimeError("INDOONE_TRUSTED_PROXY_IPS must be configured when proxy headers are trusted")
        if trust_proxy_headers and "*" in proxies:
            raise RuntimeError("INDOONE_TRUSTED_PROXY_IPS must not contain wildcard '*'")


def configure_network_security(app: FastAPI) -> None:
    """Install safe Host/CORS/proxy middleware using environment configuration."""
    validate_network_security_config()
    hosts = allowed_hosts()
    if hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    origins = allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
            max_age=600,
        )

    trust_proxy_headers = os.getenv("INDOONE_TRUST_PROXY_HEADERS", "false").strip().lower() == "true"
    if trust_proxy_headers:
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted_proxy_ips())
