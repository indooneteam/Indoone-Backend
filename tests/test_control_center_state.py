from __future__ import annotations

from app.capabilities import control_center


def test_defaults_preserve_existing_channel_reply_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_WHATSAPP_AI_REPLY_ENABLED", "true")
    monkeypatch.setenv("INDOONE_INSTAGRAM_AI_REPLY_ENABLED", "false")

    state = control_center.get_control_state()

    assert state["controls"]["global_intake_enabled"] is True
    assert state["controls"]["global_replies_enabled"] is True
    assert state["controls"]["channels"]["whatsapp"]["reply_enabled"] is True
    assert state["controls"]["channels"]["instagram"]["reply_enabled"] is False
    assert state["controls"]["channels"]["android"]["reply_enabled"] is True


def test_settings_are_persistent_and_channel_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))

    result = control_center.update_control_settings({
        "global_intake_enabled": False,
        "channels": {
            "whatsapp": {"intake_enabled": False, "reply_enabled": False},
            "instagram": {"reply_enabled": True},
        },
    })

    controls = result["controls"]
    assert controls["global_intake_enabled"] is False
    assert controls["channels"]["whatsapp"]["effective_intake_enabled"] is False
    assert controls["channels"]["whatsapp"]["effective_reply_enabled"] is False
    assert controls["channels"]["instagram"]["reply_enabled"] is True
    assert controls["channels"]["telegram"]["reply_enabled"] is False

    reloaded = control_center.get_control_state()
    assert reloaded["controls"]["global_intake_enabled"] is False
    assert reloaded["controls"]["channels"]["instagram"]["reply_enabled"] is True


def test_invalid_settings_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))

    for payload in (
        {},
        {"global_intake_enabled": "false"},
        {"unknown_setting": True},
        {"channels": {"whatsapp": {"reply_enabled": "yes"}}},
        {"channels": {"unknown": {"intake_enabled": False}}},
    ):
        try:
            control_center.update_control_settings(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid setting payload was accepted: {payload!r}")


def test_event_metrics_and_activity_are_deduplicated_and_content_free(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))

    assert control_center.record_control_event(
        "whatsapp", "request", "success", "/api/integrations/whatsapp/webhook", 200
    )
    assert control_center.record_control_event(
        "whatsapp", "reply", "success", "/api/integrations/whatsapp/webhook", 200,
        dedupe_key="reply:whatsapp:message-1",
    )
    assert not control_center.record_control_event(
        "whatsapp", "reply", "success", "/api/integrations/whatsapp/webhook", 200,
        dedupe_key="reply:whatsapp:message-1",
    )
    control_center.record_control_event(
        "instagram", "request", "blocked", "/api/integrations/instagram/webhook", 200
    )
    control_center.record_control_event(
        "telegram", "reply", "failed", "/api/telegram/webhook/incoming", 502
    )

    metrics = control_center.get_control_metrics()
    assert metrics["channels"]["whatsapp"]["requests"] == {
        "total": 1, "success": 1, "failed": 0, "blocked": 0
    }
    assert metrics["channels"]["whatsapp"]["replies"]["sent"] == 1
    assert metrics["channels"]["instagram"]["requests"]["blocked"] == 1
    assert metrics["channels"]["telegram"]["replies"]["failed"] == 1
    assert metrics["totals"]["requests"]["total"] == 2

    activity = control_center.get_control_activity()
    assert len(activity["events"]) == 4
    assert "body" not in str(activity).lower()
    assert "message_id" not in str(activity).lower()


def test_admin_token_is_separate_and_constant_time_checked(monkeypatch):
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "a" * 40)

    assert control_center.is_control_center_admin_token("a" * 40)
    assert control_center.is_control_center_admin_authorization("Bearer " + "a" * 40)
    assert not control_center.is_control_center_admin_token("b" * 40)
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "short")
    assert not control_center.is_control_center_admin_token("short")
