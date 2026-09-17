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


def test_issue_approval_requires_authentication(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "y" * 32)
    response = client.post("/api/approvals/issue", json={"tool": "phone_call"})
    assert response.status_code == 401
    assert response.json()["code"] == "HTTP_401"


def test_issue_approval_rejects_safe_tool(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "y" * 32)
    response = client.post("/api/approvals/issue", json={"tool": "calculator"}, headers=_auth_headers())
    assert response.status_code == 400
    assert response.json()["code"] == "HTTP_400"


def test_issue_approval_binds_token_to_authenticated_user(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "y" * 32)
    response = client.post(
        "/api/approvals/issue",
        json={"tool": "phone_call", "ttl_seconds": 120},
        headers=_auth_headers("user-one"),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "phone_call"
    assert payload["expires_in_seconds"] == 120
    assert payload["approval_token"]


def test_issue_approval_rejects_unknown_tool(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "y" * 32)
    response = client.post("/api/approvals/issue", json={"tool": "gmail_delete"}, headers=_auth_headers())
    assert response.status_code == 404
    assert response.json()["code"] == "HTTP_404"
