from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

from app.ai.tool_registry import get_tool_spec

MAX_APPROVAL_USER_LENGTH = 256
MAX_APPROVAL_TOOL_LENGTH = 128
MAX_APPROVAL_TOKEN_LENGTH = 2048
MAX_APPROVAL_TOKEN_COUNT = 32


@dataclass(frozen=True)
class ApprovalDecision:
    allowed: bool
    requires_approval: bool
    reason: str


def _approval_secret() -> bytes:
    secret = os.getenv("INDOONE_APPROVAL_SECRET", "").strip()
    if len(secret) < 32:
        raise RuntimeError("INDOONE_APPROVAL_SECRET must be at least 32 characters")
    return secret.encode("utf-8")


def _normalize_identity(value: str, field: str, limit: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    if len(normalized) > limit:
        raise ValueError(f"{field} is too long")
    return normalized


def issue_approval_token(user_id: str, tool: str, ttl_seconds: int = 300, now: int | None = None) -> str:
    normalized_user = _normalize_identity(user_id, "user_id", MAX_APPROVAL_USER_LENGTH)
    normalized_tool = _normalize_identity(tool, "tool", MAX_APPROVAL_TOOL_LENGTH).lower()
    spec = get_tool_spec(normalized_tool)
    if spec is None:
        raise ValueError("tool is not registered")
    if not spec.requires_approval:
        raise ValueError("tool does not require approval")
    if ttl_seconds < 1 or ttl_seconds > 3600:
        raise ValueError("approval token ttl must be between 1 and 3600 seconds")
    timestamp = int(time.time()) if now is None else int(now)
    payload = {
        "v": 1,
        "sub": normalized_user,
        "tool": spec.name,
        "exp": timestamp + ttl_seconds,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(_approval_secret(), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"v1.{encoded}.{signature}"


def validate_approval_token(token: str, user_id: str, tool: str, now: int | None = None) -> bool:
    if len(token.strip()) > MAX_APPROVAL_TOKEN_LENGTH:
        return False
    try:
        normalized_user = _normalize_identity(user_id, "user_id", MAX_APPROVAL_USER_LENGTH)
        normalized_tool = _normalize_identity(tool, "tool", MAX_APPROVAL_TOOL_LENGTH).lower()
    except ValueError:
        return False
    spec = get_tool_spec(normalized_tool)
    if spec is None or not spec.requires_approval:
        return False
    token_value = token.strip()
    parts = token_value.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return False
    encoded, signature = parts[1], parts[2]
    if not encoded or len(signature) != hashlib.sha256().digest_size * 2:
        return False
    try:
        expected = hmac.new(_approval_secret(), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    except UnicodeEncodeError:
        return False
    if not hmac.compare_digest(signature, expected):
        return False
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    timestamp = int(time.time()) if now is None else int(now)
    return (
        payload.get("v") == 1
        and payload.get("sub") == normalized_user
        and payload.get("tool") == spec.name
        and isinstance(payload.get("exp"), int)
        and timestamp < payload["exp"]
    )


def decide_tool(
    name: str,
    approved_tools: set[str] | frozenset[str] | None = None,
    approval_tokens: Iterable[str] | None = None,
    user_id: str = "",
) -> ApprovalDecision:
    del approved_tools  # Client-provided tool names are never a security boundary.
    spec = get_tool_spec(name)
    if spec is None:
        return ApprovalDecision(False, True, "tool is not registered")
    if not spec.requires_approval:
        return ApprovalDecision(True, False, "tool is auto-approved by policy")
    if not user_id.strip():
        return ApprovalDecision(False, True, "authenticated user is required for approval")
    try:
        tokens = tuple(approval_tokens or ())
        if len(tokens) > MAX_APPROVAL_TOKEN_COUNT:
            return ApprovalDecision(False, True, "too many approval tokens")
        if any(
            isinstance(token, str)
            and len(token.strip()) <= MAX_APPROVAL_TOKEN_LENGTH
            and validate_approval_token(token, user_id, spec.name)
            for token in tokens
        ):
            return ApprovalDecision(True, True, "server-issued approval token accepted")
    except RuntimeError:
        return ApprovalDecision(False, True, "approval service is not configured")
    return ApprovalDecision(False, True, "valid server-issued approval is required")
