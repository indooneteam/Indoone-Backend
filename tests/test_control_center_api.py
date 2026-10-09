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
