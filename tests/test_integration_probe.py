import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.capabilities.integrations import probe_integration
from app.capabilities.store import upsert_integration_token
from app.main import app


@pytest.mark.asyncio
async def test_provider_probe_uses_user_scoped_encrypted_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    cipher = Fernet()
    access = "user-one-secret"
    upsert_integration_token("user-one", "github", cipher.encrypt(access.encode()), None, "Bearer", "repo:read", None)

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


def test_provider_probe_endpoint_rejects_unconnected_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post("/api/integrations/github/probe", json={"user_id": "user-two"})
    assert response.status_code == 400
    assert "not connected" in response.json()["detail"]
