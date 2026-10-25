from fastapi.testclient import TestClient

from app.main import app


def test_health_includes_request_id_and_security_headers() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_request_content_length_limit(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MAX_REQUEST_BYTES", "10")
    with TestClient(app) as client:
        response = client.post("/api/chat", headers={"content-length": "11"}, json={"message": "x"})

    assert response.status_code == 413
    assert response.headers.get("X-Request-ID")
    assert response.headers["Cache-Control"] == "no-store"


def test_unhandled_exception_has_safe_error_contract() -> None:
    async def boom() -> None:
        raise RuntimeError("secret implementation detail")

    app.add_api_route("/__test_unhandled", boom, methods=["GET"])
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/__test_unhandled")
    finally:
        app.router.routes = [route for route in app.router.routes if getattr(route, "path", "") != "/__test_unhandled"]

    assert response.status_code == 500
    assert response.headers.get("X-Request-ID")
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert response.json()["message"] == "internal server error"
    assert "secret implementation detail" not in response.text
