from __future__ import annotations

import os

import uvicorn

_PRODUCTION_ENVIRONMENTS = {"prod", "production"}


def _is_production() -> bool:
    return os.getenv("INDOONE_ENV", "development").strip().lower() in _PRODUCTION_ENVIRONMENTS


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def server_settings() -> dict[str, object]:
    """Build safe Uvicorn settings for local and production execution."""
    host = os.getenv("INDOONE_SERVER_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = _bounded_int("PORT", _bounded_int("INDOONE_SERVER_PORT", 8000, 1, 65535), 1, 65535)
    workers = _bounded_int("INDOONE_SERVER_WORKERS", 1, 1, 8)
    limit_concurrency = _bounded_int("INDOONE_SERVER_LIMIT_CONCURRENCY", 200, 1, 10_000)
    timeout_keep_alive = _bounded_int("INDOONE_SERVER_TIMEOUT_KEEP_ALIVE", 5, 1, 120)
    timeout_graceful_shutdown = _bounded_int("INDOONE_SERVER_TIMEOUT_GRACEFUL_SHUTDOWN", 30, 1, 300)
    reload = os.getenv("INDOONE_SERVER_RELOAD", "false").strip().lower() == "true"

    if reload and workers != 1:
        raise RuntimeError("INDOONE_SERVER_RELOAD requires INDOONE_SERVER_WORKERS=1")
    if _is_production() and reload:
        raise RuntimeError("INDOONE_SERVER_RELOAD must be false in production")

    return {
        "app": "app.main:app",
        "host": host,
        "port": port,
        "workers": workers,
        "reload": reload,
        "limit_concurrency": limit_concurrency,
        "timeout_keep_alive": timeout_keep_alive,
        "timeout_graceful_shutdown": timeout_graceful_shutdown,
        "server_header": False,
        # app/main.py owns the trusted-proxy allowlist; do not let Uvicorn
        # trust arbitrary forwarded headers at the socket layer.
        "proxy_headers": False,
        "log_level": os.getenv("INDOONE_SERVER_LOG_LEVEL", "info").strip().lower() or "info",
    }


def main() -> None:
    uvicorn.run(**server_settings())


if __name__ == "__main__":
    main()
