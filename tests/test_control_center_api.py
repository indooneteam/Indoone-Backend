from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _headers(token: str = "control-center-admin-token-for-tests-123456789") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_control_center_routes_require_the_dedicated_admin_token(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    client = TestClient(app)

    assert client.get("/api/control-center/status").status_code == 401
    assert client.get("/api/control-center/status", headers=_headers("wrong-token")).status_code == 401
    response = client.get("/api/control-center/status", headers=_headers())

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_control_center_settings_patch_is_persisted_and_metrics_are_real(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={
            "global_intake_enabled": False,
            "channels": {
                "whatsapp": {"intake_enabled": False, "reply_enabled": False},
                "instagram": {"reply_enabled": True},
            },
        },
    )
    assert patch.status_code == 200
    controls = patch.json()["controls"]
    assert controls["global_intake_enabled"] is False
    assert controls["channels"]["whatsapp"]["effective_intake_enabled"] is False
    assert controls["channels"]["instagram"]["reply_enabled"] is True

    reloaded = client.get("/api/control-center/status", headers=headers)
    assert reloaded.status_code == 200
    assert reloaded.json()["controls"]["global_intake_enabled"] is False

    metrics = client.get("/api/control-center/metrics", headers=headers)
    activity = client.get("/api/control-center/activity?limit=10", headers=headers)
    assert metrics.status_code == 200
    assert metrics.json()["channels"]["whatsapp"]["requests"]["total"] == 0
    assert activity.status_code == 200
    assert activity.json()["events"] == []


def test_control_center_rejects_invalid_settings_and_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    client = TestClient(app)
    headers = _headers()

    invalid_patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"global_intake_enabled": "off"},
    )
    invalid_metrics = client.get("/api/control-center/metrics?window_hours=0", headers=headers)
    invalid_activity = client.get("/api/control-center/activity?limit=500", headers=headers)

    assert invalid_patch.status_code == 422
    assert invalid_metrics.status_code == 422
    assert invalid_activity.status_code == 422


def test_global_intake_pause_blocks_app_routes_but_leaves_health_and_admin_online(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch("/api/control-center/settings", headers=headers, json={"global_intake_enabled": False})
    assert patch.status_code == 200

    blocked = client.get("/api/capabilities")
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "INTAKE_PAUSED"

    # The server and authenticated control plane remain available while the app intake is paused.
    assert client.get("/health").status_code == 200
    status = client.get("/api/control-center/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["controls"]["global_intake_enabled"] is False

    metrics = client.get("/api/control-center/metrics", headers=headers)
    assert metrics.status_code == 200
    assert metrics.json()["channels"]["android"]["requests"]["blocked"] == 1

    resumed = client.patch("/api/control-center/settings", headers=headers, json={"global_intake_enabled": True})
    assert resumed.status_code == 200
    assert client.get("/api/capabilities").status_code == 200


def test_signed_whatsapp_webhook_is_acknowledged_but_not_processed_while_intake_paused(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json

    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    monkeypatch.setenv("INDOONE_WHATSAPP_APP_SECRET", "whatsapp-test-app-secret")
    client = TestClient(app)
    headers = _headers()

    paused = client.patch("/api/control-center/settings", headers=headers, json={"global_intake_enabled": False})
    assert paused.status_code == 200

    body = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode()
    signature = hmac.new(b"whatsapp-test-app-secret", body, hashlib.sha256).hexdigest()
    response = client.post(
        "/api/integrations/whatsapp/webhook",
        content=body,
        headers={"X-Hub-Signature-256": "sha256=" + signature},
    )

    assert response.status_code == 200
    assert response.json()["received"] is True
    assert response.json()["processed"] is False
    assert response.json()["reason"] == "intake_paused"

    metrics = client.get("/api/control-center/metrics", headers=headers)
    assert metrics.status_code == 200
    assert metrics.json()["channels"]["whatsapp"]["requests"]["blocked"] == 1


def test_individual_android_intake_pause_does_not_pause_other_control_plane_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"channels": {"android": {"intake_enabled": False}}},
    )
    assert patch.status_code == 200

    blocked = client.get("/api/capabilities")
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "INTAKE_PAUSED"

    still_online = client.get("/api/control-center/status", headers=headers)
    assert still_online.status_code == 200
    assert still_online.json()["controls"]["channels"]["whatsapp"]["effective_intake_enabled"] is True



def test_android_reply_off_skips_model_and_tracks_skipped_reply(tmp_path, monkeypatch):
    import app.api.chat as chat_api

    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    monkeypatch.setattr(chat_api, "current_user_id", lambda request: "test-user")

    async def unexpected_generate(*args, **kwargs):
        raise AssertionError("Android model generation must not happen when replies are disabled")

    monkeypatch.setattr(chat_api, "generate_reply", unexpected_generate)
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"global_replies_enabled": False},
    )
    assert patch.status_code == 200

    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "paused" in response.json()["message"].lower()

    metrics = client.get("/api/control-center/metrics", headers=headers)
    assert metrics.status_code == 200
    assert metrics.json()["channels"]["android"]["replies"]["skipped"] == 1



def test_whatsapp_intake_pause_is_independent_of_other_channels(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json

    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    monkeypatch.setenv("INDOONE_WHATSAPP_APP_SECRET", "whatsapp-test-app-secret")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"global_intake_enabled": True, "channels": {"whatsapp": {"intake_enabled": False}}},
    )
    assert patch.status_code == 200

    body = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode()
    signature = hmac.new(b"whatsapp-test-app-secret", body, hashlib.sha256).hexdigest()
    response = client.post(
        "/api/integrations/whatsapp/webhook",
        content=body,
        headers={"X-Hub-Signature-256": "sha256=" + signature},
    )
    assert response.status_code == 200
    assert response.json()["reason"] == "intake_paused"

    state = client.get("/api/control-center/status", headers=headers).json()["controls"]
    assert state["global_intake_enabled"] is True
    assert state["channels"]["whatsapp"]["effective_intake_enabled"] is False
    assert state["channels"]["instagram"]["effective_intake_enabled"] is True

    metrics = client.get("/api/control-center/metrics", headers=headers).json()
    assert metrics["channels"]["whatsapp"]["requests"]["blocked"] == 1


def test_instagram_intake_pause_is_independent_of_whatsapp(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json

    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    monkeypatch.setenv("INDOONE_INSTAGRAM_APP_SECRET", "instagram-test-app-secret")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"channels": {"instagram": {"intake_enabled": False}}},
    )
    assert patch.status_code == 200

    body = json.dumps({"object": "instagram", "entry": []}).encode()
    signature = hmac.new(b"instagram-test-app-secret", body, hashlib.sha256).hexdigest()
    response = client.post(
        "/api/integrations/instagram/webhook",
        content=body,
        headers={"X-Hub-Signature-256": "sha256=" + signature},
    )
    assert response.status_code == 200
    assert response.json()["reason"] == "intake_paused"

    state = client.get("/api/control-center/status", headers=headers).json()["controls"]
    assert state["global_intake_enabled"] is True
    assert state["channels"]["instagram"]["effective_intake_enabled"] is False
    assert state["channels"]["whatsapp"]["effective_intake_enabled"] is True


def test_telegram_intake_pause_acknowledges_authenticated_update_without_processing(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "false")
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite3"))
    monkeypatch.setenv("INDOONE_CONTROL_CENTER_ADMIN_TOKEN", "control-center-admin-token-for-tests-123456789")
    monkeypatch.setenv("INDOONE_TELEGRAM_WEBHOOK_SECRET", "telegram-test-webhook-secret")
    client = TestClient(app)
    headers = _headers()

    patch = client.patch(
        "/api/control-center/settings",
        headers=headers,
        json={"channels": {"telegram": {"intake_enabled": False}}},
    )
    assert patch.status_code == 200

    response = client.post(
        "/api/telegram/webhook/incoming",
        json={"update_id": 54321, "message": {"chat": {"id": 111}, "text": "hello"}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-test-webhook-secret"},
    )
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert response.json()["processed"] is False
    assert response.json()["reason"] == "intake_paused"

    state = client.get("/api/control-center/status", headers=headers).json()["controls"]
    assert state["global_intake_enabled"] is True
    assert state["channels"]["telegram"]["effective_intake_enabled"] is False
    assert state["channels"]["whatsapp"]["effective_intake_enabled"] is True

    metrics = client.get("/api/control-center/metrics", headers=headers).json()
    assert metrics["channels"]["telegram"]["requests"]["blocked"] == 1
