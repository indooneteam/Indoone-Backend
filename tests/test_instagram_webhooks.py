import hashlib
import hmac
import json

import pytest

from app.capabilities import instagram_webhooks


def test_verify_challenge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_INSTAGRAM_WEBHOOK_VERIFY_TOKEN", "verify-secret")
    assert instagram_webhooks.verify_challenge("subscribe", "12345", "verify-secret") == "12345"
    with pytest.raises(PermissionError):
        instagram_webhooks.verify_challenge("subscribe", "12345", "wrong")
    with pytest.raises(ValueError):
        instagram_webhooks.verify_challenge("bad", "12345", "verify-secret")


def test_verify_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_SECRET", "app-secret")
    body = b'{"object":"instagram"}'
    digest = hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()
    instagram_webhooks.verify_signature(body, f"sha256={digest}")
    with pytest.raises(PermissionError):
        instagram_webhooks.verify_signature(body, "sha256=bad")
    with pytest.raises(PermissionError):
        instagram_webhooks.verify_signature(body, "md5=bad")


def test_normalize_event_is_secret_safe() -> None:
    payload = {
        "object": "instagram",
        "entry": [
            {"id": "acct-1", "time": 123, "changes": [{"field": "comments"}], "messaging": [{"sender": {"id": "u1"}}]},
            {"id": "acct-2", "changes": []},
        ],
    }
    result = instagram_webhooks.normalize_event(payload)
    assert result["integration"] == "instagram"
    assert result["event_count"] == 2
    assert len(result["event_id"]) == 64
    assert result["secrets_exposed"] is False
    assert result["entries"][0]["id"] == "acct-1"
    assert result["entries"][0]["changes"][0]["field"] == "comments"


def test_event_id_is_deterministic() -> None:
    payload = {"object": "instagram", "entry": [{"id": "1"}]}
    first = instagram_webhooks.normalize_event(payload)
    second = instagram_webhooks.normalize_event(json.loads(json.dumps(payload)))
    assert first["event_id"] == second["event_id"]


def test_invalid_object_is_rejected() -> None:
    with pytest.raises(ValueError):
        instagram_webhooks.normalize_event({"object": "facebook", "entry": []})


def test_entry_event_counts_are_bounded() -> None:
    payload = {"object": "instagram", "entry": [{"changes": [{}] * 101}]}
    with pytest.raises(ValueError, match="too many events"):
        instagram_webhooks.normalize_event(payload)


def test_entry_count_is_bounded() -> None:
    payload = {"object": "instagram", "entry": [{}] * 101}
    with pytest.raises(ValueError, match="too many entries"):
        instagram_webhooks.normalize_event(payload)
