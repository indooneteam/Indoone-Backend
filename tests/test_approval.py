from __future__ import annotations

import os
import time

import pytest

from app.ai.approval import issue_approval_token, validate_approval_token


SECRET = "b" * 32


def test_approval_token_is_bound_to_user_and_tool(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    token = issue_approval_token("user-1", "phone_call_contact", ttl_seconds=60, now=1_000)
    assert validate_approval_token(token, "user-1", "phone_call_contact", now=1_001)
    assert not validate_approval_token(token, "user-2", "phone_call_contact", now=1_001)
    assert not validate_approval_token(token, "gmail_delete", now=1_001)


def test_approval_token_expires(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    token = issue_approval_token("user-1", "phone_call_contact", ttl_seconds=10, now=1_000)
    assert validate_approval_token(token, "user-1", "phone_call_contact", now=1_009)
    assert not validate_approval_token(token, "user-1", "phone_call_contact", now=1_010)


def test_approval_token_rejects_tampering(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    token = issue_approval_token("user-1", "phone_call_contact", now=int(time.time()))
    prefix, payload, signature = token.split(".")
    tampered = f"{prefix}.{payload}x.{signature}"
    assert not validate_approval_token(tampered, "user-1", "phone_call_contact")


def test_approval_token_requires_server_secret(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_APPROVAL_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="INDOONE_APPROVAL_SECRET"):
        issue_approval_token("user-1", "phone_call_contact")
