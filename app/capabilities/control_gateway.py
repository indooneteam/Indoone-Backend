from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


async def send_via_control_gateway(
    channel: str,
    target_url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    *,
    timeout: float = 30.0,
) -> httpx.Response | None:
    """Send a provider message via the gateway, or return None for explicit legacy mode.

    Production should set INDOONE_GATEWAY_REQUIRED=true after gateway deployment. In that
    mode, missing or invalid gateway configuration fails closed and never falls back to the
    provider API directly.
    """
    if channel not in {"whatsapp", "instagram", "telegram"}:
        raise ValueError("unsupported control-gateway channel")
    origin = os.getenv("INDOONE_CONTROL_GATEWAY_ORIGIN", "").strip().rstrip("/")
    required = _enabled("INDOONE_GATEWAY_REQUIRED")
    if not origin:
        if required:
            raise RuntimeError("Control Gateway is required but its origin is not configured")
        return None

    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("Control Gateway origin is invalid") from exc
    valid_common = bool(
        parsed.hostname
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )
    valid_https = parsed.scheme == "https" and port in {None, 443}
    valid_loopback = (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "::1"}
        and port == 8080
    )
    if not valid_common or not (valid_https or valid_loopback):
        raise RuntimeError(
            "Control Gateway origin must be HTTPS or local loopback HTTP on port 8080"
        )

    service_token = os.getenv("INDOONE_GATEWAY_BACKEND_TOKEN", "").strip()
    if len(service_token) < 32:
        raise RuntimeError("Control Gateway service token is missing or too short")

    request_url = f"{origin}/internal/egress/{channel}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(
                request_url,
                headers={"Authorization": f"Bearer {service_token}"},
                json={
                    "target_url": target_url,
                    "headers": headers,
                    "json": payload,
                },
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("Control Gateway egress request failed") from exc

    if response.status_code == 423:
        raise PermissionError(f"{channel} replies are paused by the Control Gateway")
    if response.status_code == 401:
        try:
            detail = response.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""
        if detail == "gateway backend service authorization required":
            raise RuntimeError("Control Gateway rejected backend service authentication")
    return response



async def check_reply_allowed(channel: str) -> bool | None:
    """Query persistent gateway reply state before spending AI tokens.

    Returns None only in explicit legacy mode (no gateway origin and gateway not required).
    If a configured gateway is unreachable or rejects authentication, fail closed.
    """
    if channel not in {"whatsapp", "instagram", "telegram", "android"}:
        raise ValueError("unsupported control-gateway channel")
    origin = os.getenv("INDOONE_CONTROL_GATEWAY_ORIGIN", "").strip().rstrip("/")
    required = _enabled("INDOONE_GATEWAY_REQUIRED")
    if not origin:
        if required:
            raise RuntimeError("Control Gateway is required but its origin is not configured")
        return None

    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("Control Gateway origin is invalid") from exc
    valid_common = bool(
        parsed.hostname
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )
    valid_https = parsed.scheme == "https" and port in {None, 443}
    valid_loopback = (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "::1"}
        and port == 8080
    )
    if not valid_common or not (valid_https or valid_loopback):
        raise RuntimeError(
            "Control Gateway origin must be HTTPS or local loopback HTTP on port 8080"
        )

    service_token = os.getenv("INDOONE_GATEWAY_BACKEND_TOKEN", "").strip()
    if len(service_token) < 32:
        raise RuntimeError("Control Gateway service token is missing or too short")
    url = f"{origin}/internal/replies/check/{channel}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.post(url, headers={"Authorization": f"Bearer {service_token}"})
    except httpx.HTTPError as exc:
        raise RuntimeError("Control Gateway reply-state check failed") from exc
    if response.status_code != 200:
        raise RuntimeError("Control Gateway rejected the reply-state check")
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("Control Gateway returned an invalid reply-state response") from exc
    allowed = body.get("allowed") if isinstance(body, dict) else None
    if body.get("status") != "ok" or type(allowed) is not bool:
        raise RuntimeError("Control Gateway returned an invalid reply-state response")
    return allowed
