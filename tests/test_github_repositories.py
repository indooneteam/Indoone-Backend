import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.capabilities.integrations import list_github_repositories
from app.capabilities.store import upsert_integration_token
from app.main import app


@pytest.mark.asyncio
async def test_list_github_repositories_uses_user_scoped_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    access = "github-secret"
    upsert_integration_token("user-one", "github", cipher.encrypt(access.encode()), None, "Bearer", "repo:read", None)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, object]]:
            return [{
                "id": 10,
                "name": "indoone",
                "full_name": "indooneteam/indoone",
                "private": True,
                "html_url": "https://github.com/indooneteam/indoone",
                "default_branch": "develop",
                "description": "backend",
                "ignored_secret": "must not escape",
            }]

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, **kwargs):
            assert url == "https://api.github.com/user/repos"
            assert kwargs["headers"]["Authorization"] == f"Bearer {access}"
            assert kwargs["params"] == {"page": 2, "per_page": 10, "sort": "updated", "direction": "desc"}
            return Response()

    monkeypatch.setattr("app.capabilities.integrations.httpx.AsyncClient", Client)
    result = await list_github_repositories("user-one", page=2, per_page=10)
    assert result["repositories"] == [{
        "id": 10,
        "name": "indoone",
        "full_name": "indooneteam/indoone",
        "private": True,
        "html_url": "https://github.com/indooneteam/indoone",
        "default_branch": "develop",
        "description": "backend",
    }]
    assert result["secrets_exposed"] is False

    with pytest.raises(ValueError, match="not connected"):
        await list_github_repositories("user-two")


def test_github_repositories_endpoint_validates_unconnected_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post("/api/integrations/github/repositories", json={"user_id": "user-two"})
    assert response.status_code == 400
    assert "not connected" in response.json()["message"]
