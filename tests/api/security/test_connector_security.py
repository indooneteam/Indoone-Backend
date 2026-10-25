import base64
import hashlib
import hmac
import time

from fastapi.testclient import TestClient

from app.ai.execution.connector_authorization import scope_allows
from app.main import app


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _token(user_id: str) -> str:
    encoded_user = _encode(user_id.encode("utf-8"))
    encoded_timestamp = _encode(str(int(time.time())).encode("ascii"))
    payload = f"{encoded_user}.{encoded_timestamp}".encode("ascii")
    signature = _encode(hmac.new(b"x" * 32, payload, hashlib.sha256).digest())
    return f"Bearer {encoded_user}.{encoded_timestamp}.{signature}"


def test_connector_user_scope_blocks_cross_user_request(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/repositories",
            headers={"Authorization": _token("user-one")},
            json={"user_id": "user-two"},
        )

    assert response.status_code == 403
    payload = response.json()
    assert payload["code"] == "HTTP_403"
    assert payload["message"] == "user scope mismatch"
    assert "request_id" in payload


def test_connector_user_scope_blocks_query_user_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/repositories?user_id=user-two",
            headers={"Authorization": _token("user-one")},
            json={},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "HTTP_403"


def test_connector_user_scope_ignores_non_connector_prefix(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations-evil/github/repositories",
            headers={"Authorization": _token("user-one")},
            json={"user_id": "user-two"},
        )

    assert response.status_code in {404, 405, 422}


def test_connector_user_scope_accepts_matching_authenticated_user(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/repositories",
            headers={"Authorization": _token("user-one")},
            json={"user_id": "user-one"},
        )

    assert response.status_code in {200, 400, 502, 503}


def test_connector_user_scope_requires_auth_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")
    monkeypatch.delenv("INDOONE_AUTH_SECRET", raising=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/repositories",
            json={"user_id": "user-one"},
        )

    assert response.status_code == 401
    payload = response.json()
    assert payload["code"] == "HTTP_401"
    assert payload["message"] == "authenticated connector user required"
    assert "request_id" in payload



def test_github_public_repo_scope_allows_read_only_capabilities() -> None:
    assert scope_allows("github", "public_repo read:user", "repos.read") is True
    assert scope_allows("github", "public_repo read:user", "issues.read") is True
    assert scope_allows("github", "public_repo read:user", "pull_requests.read") is True
    assert scope_allows("github", "public_repo read:user", "issues.write") is False
    assert scope_allows("github", "public_repo read:user", "comments.write") is False


def test_connector_user_scope_rejects_body_user_behind_matching_query(monkeypatch) -> None:
    """A matching query string must not authorise a different body user_id.

    The handler reads the user it acts for from the body. This request names the caller in
    the query string, where nothing reads it, and somebody else in the body, where the
    handler does - so it must be refused. Checking only the query let it through.
    """
    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_CONNECTOR_AUTH_REQUIRED", "true")

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/repositories?user_id=user-one",
            headers={"Authorization": _token("user-one")},
            json={"user_id": "user-two"},
        )

    assert response.status_code == 403
    assert response.json()["message"] == "user scope mismatch"


