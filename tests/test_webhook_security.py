import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app.capabilities.telegram import validate_webhook_secret
from app.capabilities.whatsapp import validate_signature
from app.main import app


def test_telegram_webhook_secret_is_required() -> None:
    assert validate_webhook_secret(None) is False
    assert validate_webhook_secret("wrong") is False


def test_whatsapp_signature_rejects_missing_or_invalid_header() -> None:
    assert validate_signature("app-secret", None, b"{}") is False
    assert validate_signature("app-secret", "sha256=bad", b"{}") is False


def test_whatsapp_signature_accepts_exact_body_signature() -> None:
    raw = b'{"entry":[]}'
    digest = hmac.new(b"app-secret", raw, hashlib.sha256).hexdigest()

    assert validate_signature("app-secret", f"sha256={digest}", raw) is True


def test_whatsapp_webhook_fails_closed_without_app_secret(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_WHATSAPP_APP_SECRET", raising=False)
    response = TestClient(app).post("/api/integrations/whatsapp/webhook", json={"entry": []})

    assert response.status_code == 503
    assert "signature validation is not configured" in response.json()["message"]


def test_whatsapp_webhook_requires_valid_signature(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_WHATSAPP_APP_SECRET", "app-secret")
    payload = {"entry": []}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    response = TestClient(app).post(
        "/api/integrations/whatsapp/webhook",
        content=raw,
        headers={"content-type": "application/json", "X-Hub-Signature-256": "sha256=bad"},
    )

    assert response.status_code == 403


def test_whatsapp_webhook_accepts_valid_signature(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_WHATSAPP_APP_SECRET", "app-secret")
    raw = json.dumps({"entry": []}, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(b"app-secret", raw, hashlib.sha256).hexdigest()

    response = TestClient(app).post(
        "/api/integrations/whatsapp/webhook",
        content=raw,
        headers={"content-type": "application/json", "X-Hub-Signature-256": f"sha256={digest}"},
    )

    assert response.status_code == 200
    assert response.json()["received"] is True
