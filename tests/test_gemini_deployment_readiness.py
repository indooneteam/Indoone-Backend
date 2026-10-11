from __future__ import annotations

from fastapi.testclient import TestClient

from app.ai import service
from app.main import app


client = TestClient(app, raise_server_exceptions=False)


def test_local_provider_status_does_not_require_gemini_key(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    status = service.ai_provider_status()

    assert status == {
        "provider": "local",
        "model": None,
        "configuration_ready": True,
        "api_key_configured": None,
    }


def test_gemini_provider_status_reports_selected_model_without_secret(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")

    status = service.ai_provider_status()

    assert status == {
        "provider": "gemini",
        "model": "gemma-4-26b-a4b-it",
        "configuration_ready": True,
        "api_key_configured": True,
    }
    assert "secret-test-key" not in repr(status)


def test_gemini_provider_status_uses_supported_default_model(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")

    status = service.ai_provider_status()

    assert status["provider"] == "gemini"
    assert status["model"] == "gemma-4-26b-a4b-it"
    assert status["configuration_ready"] is True


def test_ready_fails_when_gemini_backend_has_no_api_key(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["code"] == "AI_PROVIDER_NOT_READY"
    assert response.headers["X-Request-ID"]


def test_ready_accepts_configured_gemini_backend(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_details_reports_provider_and_model_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

    response = client.get("/health/details")

    assert response.status_code == 200
    assert response.json()["ai"]["provider"] == "gemini"
    assert response.json()["ai"]["model"] == "gemma-4-26b-a4b-it"
    assert response.json()["ai"]["configuration_ready"] is True
    assert response.json()["ai"]["api_key_configured"] is True
    assert "secret-test-key" not in response.text
