import base64
import hashlib
import hmac
import time

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _auth_headers(user_id: str = "user-one") -> dict[str, str]:
    user = _encode(user_id.encode("utf-8"))
    timestamp = _encode(str(int(time.time())).encode("ascii"))
    payload = f"{user}.{timestamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{timestamp}.{signature}"}


def test_document_analysis_requires_authentication(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    response = client.post(
        "/api/documents/analyze",
        json={
            "filename": "broken.pdf",
            "mime_type": "application/pdf",
            "content_base64": _encode(b"not a pdf"),
        },
    )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


def test_document_batch_size_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    oversized = "A" * 30_000_001
    response = client.post(
        "/api/documents/analyze-batch",
        json={
            "documents": [
                {
                    "filename": "large.pdf",
                    "mime_type": "application/pdf",
                    "content_base64": oversized,
                }
            ]
        },
        headers=_auth_headers(),
    )

    assert response.status_code == 413
    assert response.json()["code"] == "HTTP_413"
