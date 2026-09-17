from __future__ import annotations

import time

import pytest

from app.ai.approval import decide_tool, issue_approval_token, validate_approval_token


SECRET = "b" * 32


def test_approval_token_is_bound_to_user_and_tool(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    token = issue_approval_token("user-1", "phone_call_contact", ttl_seconds=60, now=1_000)
    assert validate_approval_token(token, "user-1", "phone_call_contact", now=1_001)
    assert not validate_approval_token(token, "user-2", "phone_call_contact", now=1_001)
    assert not validate_approval_token(token, "user-1", "gmail_delete", now=1_001)


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


def test_approval_token_normalizes_tool_and_requires_approval(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    token = issue_approval_token("user-1", " PHONE_CALL_CONTACT ", ttl_seconds=60, now=1_000)
    assert validate_approval_token(token, "user-1", "phone_call_contact", now=1_001)
    with pytest.raises(ValueError, match="does not require approval"):
        issue_approval_token("user-1", " calculator ", ttl_seconds=60, now=1_000)


def test_approval_token_rejects_oversized_identity_and_token(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    with pytest.raises(ValueError, match="user_id is too long"):
        issue_approval_token("u" * 257, "phone_call_contact", now=1_000)
    with pytest.raises(ValueError, match="tool is too long"):
        issue_approval_token("user-1", "t" * 129, now=1_000)
    assert not validate_approval_token("x" * 2049, "user-1", "phone_call_contact", now=1_001)


def test_decide_tool_caps_approval_tokens(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", SECRET)
    assert not decide_tool(
        "phone_call_contact",
        approval_tokens=("bad-token",) * 33,
        user_id="user-1",
    ).allowed
