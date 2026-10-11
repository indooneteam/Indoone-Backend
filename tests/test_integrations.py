import pytest
from fastapi.testclient import TestClient

from app.capabilities.integrations import build_oauth_authorization
from app.main import app


def test_integrations_list_is_safe_metadata(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("INDOONE_SLACK_CLIENT_ID", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/integrations")
    assert response.status_code == 200
    items = response.json()["integrations"]
    assert {item["id"] for item in items} >= {"github", "google_calendar", "gmail", "slack"}
    assert all(item["configured"] is False for item in items)
    assert all("token" not in item and "secret" not in item for item in items)


def test_integration_lookup_and_missing_provider() -> None:
    with TestClient(app) as client:
        response = client.get("/api/integrations/github")
        missing = client.get("/api/integrations/not-configured")
    assert response.status_code == 200
    assert response.json()["integration"]["auth_type"] == "oauth2"
    assert "repo:read" in response.json()["integration"]["permissions"]
    assert missing.status_code == 404


def test_gmail_oauth_contract_uses_google_credentials_and_scope(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "google-client-123")
    result = build_oauth_authorization("gmail", "gmail-state-1234567890", "https://app.example/callback")
    assert "client_id=google-client-123" in result["authorization_url"]
    assert "gmail.modify" in result["authorization_url"]
    assert result["token_storage"] == "user-scoped-encrypted-server-side"
    assert result["secrets_exposed"] is False


def test_oauth_contract_requires_client_id(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_GITHUB_CLIENT_ID", raising=False)
    try:
        build_oauth_authorization("github", "a" * 32, "https://app.example/callback")
    except RuntimeError as exc:
        assert "client id" in str(exc)
    else:
        raise AssertionError("missing client id must fail")


def test_oauth_contract_builds_user_scoped_authorization(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    result = build_oauth_authorization("github", "state-token-1234567890", "https://app.example/callback")
    assert "client_id=client-123" in result["authorization_url"]
    assert "state=state-token-1234567890" in result["authorization_url"]
    assert result["token_storage"] == "user-scoped-encrypted-server-side"
    assert result["secrets_exposed"] is False


def test_integration_connect_endpoint_creates_user_bound_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_GITHUB_CLIENT_ID", "client-123")
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/connect",
            json={"user_id": "user-123", "redirect_uri": "https://app.example/callback"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["integration"] == "github"
    assert payload["state_required"] is True
    assert payload["secrets_exposed"] is False
    assert len(payload["state"]) >= 32


def test_integration_connect_endpoint_rejects_unknown_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/unknown/connect",
            json={"user_id": "user-123", "redirect_uri": "https://app.example/callback"},
        )
    assert response.status_code == 400


def test_integration_callback_requires_code_and_state() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/github/callback",
            json={"state": "short"},
        )
    assert response.status_code == 422


def test_integration_status_is_user_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.get("/api/integrations/github/status", params={"user_id": "user-123"})
    assert response.status_code == 200
    assert response.json()["connected"] is False
    assert response.json()["metadata"] is None



@pytest.mark.asyncio
async def test_expired_gmail_oauth_token_is_refreshed_and_stored_encrypted(
    monkeypatch, tmp_path
) -> None:
    from datetime import datetime, timedelta, timezone
    from cryptography.fernet import Fernet

    from app.capabilities import integrations as integration_module
    from app.capabilities.store import get_integration_token, upsert_integration_token

    key = Fernet.generate_key()
    cipher = Fernet(key)
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key.decode("ascii"))
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "google-client")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "google-secret")
    old_expiry = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    upsert_integration_token(
        "gmail-user",
        "gmail",
        cipher.encrypt(b"old-access-token"),
        cipher.encrypt(b"long-lived-refresh-token"),
        "Bearer",
        "https://www.googleapis.com/auth/gmail.modify",
        old_expiry,
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "access_token": "new-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/gmail.modify",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.request = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, data, headers):
            self.request = (url, data, headers)
            assert data["grant_type"] == "refresh_token"
            assert data["refresh_token"] == "long-lived-refresh-token"
            return FakeResponse()

    monkeypatch.setattr(integration_module.httpx, "AsyncClient", FakeAsyncClient)

    result = await integration_module.get_valid_access_token("gmail-user", "gmail")

    row = get_integration_token("gmail-user", "gmail")
    assert result == "new-access-token"
    assert row is not None
    assert cipher.decrypt(bytes(row["access_token"])) == b"new-access-token"
    assert cipher.decrypt(bytes(row["refresh_token"])) == b"long-lived-refresh-token"
    assert row["expires_at"] is not None


@pytest.mark.asyncio
async def test_expired_native_gmail_token_without_refresh_requires_reconnect(
    monkeypatch, tmp_path
) -> None:
    from datetime import datetime, timedelta, timezone
    from cryptography.fernet import Fernet

    from app.capabilities import integrations as integration_module
    from app.capabilities.store import upsert_integration_token

    key = Fernet.generate_key()
    cipher = Fernet(key)
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key.decode("ascii"))
    old_expiry = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    upsert_integration_token(
        "gmail-native-user",
        "gmail",
        cipher.encrypt(b"native-access-token"),
        None,
        "Bearer",
        "https://www.googleapis.com/auth/gmail.modify",
        old_expiry,
    )

    with pytest.raises(RuntimeError, match="reconnect gmail"):
        await integration_module.get_valid_access_token("gmail-native-user", "gmail")



def test_gmail_auth_config_exposes_only_public_client_id(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "public-google-client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "do-not-return-this-secret")
    with TestClient(app) as client:
        response = client.get("/api/integrations/gmail/auth-config")
    assert response.status_code == 200
    assert response.json()["server_client_id"] == "public-google-client-id"
    assert "client_secret" not in response.json()
    assert "do-not-return-this-secret" not in response.text


@pytest.mark.asyncio
async def test_gmail_server_auth_code_stores_refreshable_tokens_encrypted(
    monkeypatch, tmp_path
) -> None:
    from cryptography.fernet import Fernet
    from app.capabilities import integrations as integration_module
    from app.capabilities.store import get_integration_token

    key = Fernet.generate_key()
    cipher = Fernet(key)
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key.decode("ascii"))
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "google-client")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "google-secret")

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, data, headers):
            assert url == integration_module._TOKEN_URLS["gmail"]
            assert data["grant_type"] == "authorization_code"
            assert data["code"] == "server-code-once"
            return FakeResponse({
                "access_token": "short-lived-access",
                "refresh_token": "long-lived-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/gmail.modify",
            })

        async def get(self, url, *, headers):
            assert url == integration_module._PROBE_URLS["gmail"]
            assert headers["Authorization"] == "Bearer short-lived-access"
            return FakeResponse({"emailAddress": "creator@example.com"})

    monkeypatch.setattr(integration_module.httpx, "AsyncClient", FakeAsyncClient)

    result = await integration_module.exchange_gmail_server_auth_code(
        "gmail-user",
        "server-code-once",
        "https://www.googleapis.com/auth/gmail.modify",
    )

    row = get_integration_token("gmail-user", "gmail")
    assert result["connected"] is True
    assert result["summary"]["email_address"] == "creator@example.com"
    assert result["offline_access"] is True
    assert row is not None
    assert cipher.decrypt(bytes(row["access_token"])) == b"short-lived-access"
    assert cipher.decrypt(bytes(row["refresh_token"])) == b"long-lived-refresh"


@pytest.mark.asyncio
async def test_gmail_server_auth_code_fails_if_google_omits_initial_refresh_token(
    monkeypatch, tmp_path
) -> None:
    from cryptography.fernet import Fernet
    from app.capabilities import integrations as integration_module
    from app.capabilities.store import get_integration_token

    key = Fernet.generate_key()
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key.decode("ascii"))
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "google-client")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "google-secret")

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, data, headers):
            return FakeResponse({
                "access_token": "short-lived-access",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/gmail.modify",
            })

    monkeypatch.setattr(integration_module.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match="did not issue a refresh token"):
        await integration_module.exchange_gmail_server_auth_code("gmail-user", "server-code-once")

    assert get_integration_token("gmail-user", "gmail") is None
