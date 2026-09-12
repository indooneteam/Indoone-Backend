import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.capabilities.integrations import list_github_issues, list_github_pull_requests, probe_integration
from app.capabilities.store import upsert_integration_token
from app.main import app


def _setup_token(monkeypatch, tmp_path, user_id: str = "user-one") -> str:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    access = "user-one-secret"
    upsert_integration_token(user_id, "github", cipher.encrypt(access.encode()), None, "Bearer", "repo:read", None)
    return access


@pytest.mark.asyncio
async def test_provider_probe_uses_user_scoped_encrypted_token(monkeypatch, tmp_path) -> None:
    access = _setup_token(monkeypatch, tmp_path)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"login": "one", "name": "User One", "id": 1}

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, *args, **kwargs):
            assert kwargs["headers"]["Authorization"] == f"Bearer {access}"
            return Response()

    monkeypatch.setattr("app.capabilities.integrations.httpx.AsyncClient", Client)
    result = await probe_integration("user-one", "github")
    assert result["provider_ok"] is True
    assert result["summary"]["login"] == "one"
    with pytest.raises(ValueError, match="not connected"):
        await probe_integration("user-two", "github")


@pytest.mark.asyncio
async def test_github_issues_use_fixed_endpoint_and_redact_token(monkeypatch, tmp_path) -> None:
    access = _setup_token(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [{"id": 10, "number": 4, "title": "Bug", "state": "open", "html_url": "https://github.com/o/r/issues/4", "repository_url": "https://api.github.com/repos/o/r", "labels": [{"name": "bug"}]}]

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, **kwargs):
            captured.update({"url": url, **kwargs})
            return Response()

    monkeypatch.setattr("app.capabilities.integrations.httpx.AsyncClient", Client)
    result = await list_github_issues("user-one")
    assert captured["url"] == "https://api.github.com/issues"
    assert captured["headers"]["Authorization"] == f"Bearer {access}"
    assert result["issues"][0]["title"] == "Bug"
    assert result["secrets_exposed"] is False
    assert access not in str(result)


@pytest.mark.asyncio
async def test_github_pull_requests_validate_repository_and_use_fixed_host(monkeypatch, tmp_path) -> None:
    access = _setup_token(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    with pytest.raises(ValueError, match="owner/name"):
        await list_github_pull_requests("user-one", "https://evil.example/x")

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [{"id": 20, "number": 7, "title": "Feature", "state": "open", "html_url": "https://github.com/o/r/pull/7", "draft": False}]

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, **kwargs):
            captured.update({"url": url, **kwargs})
            return Response()

    monkeypatch.setattr("app.capabilities.integrations.httpx.AsyncClient", Client)
    result = await list_github_pull_requests("user-one", "o/r")
    assert captured["url"] == "https://api.github.com/repos/o/r/pulls"
    assert captured["headers"]["Authorization"] == f"Bearer {access}"
    assert result["pull_requests"][0]["number"] == 7
    assert result["secrets_exposed"] is False


def test_provider_probe_endpoint_rejects_unconnected_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post("/api/integrations/github/probe", json={"user_id": "user-two"})
    assert response.status_code == 400
    assert "not connected" in response.json()["detail"]


def test_github_pull_request_endpoint_rejects_bad_repository(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post("/api/integrations/github/pull-requests", json={"user_id": "user-one", "repository": "bad"})
    assert response.status_code == 400
    assert "owner/name" in response.json()["detail"]
