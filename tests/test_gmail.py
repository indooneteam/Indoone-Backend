import pytest
from cryptography.fernet import Fernet

from app.capabilities.gmail import get_gmail_message, list_gmail_messages, send_gmail_message
from app.capabilities.store import upsert_integration_token


@pytest.mark.asyncio
async def test_gmail_message_list_uses_user_scoped_encrypted_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    access = "gmail-secret"
    upsert_integration_token("user-one", "gmail", cipher.encrypt(access.encode()), None, "Bearer", "gmail.modify", None)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"messages": [{"id": "m1", "threadId": "t1"}], "nextPageToken": "next"}

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, **kwargs):
            assert url.endswith("/messages")
            assert kwargs["headers"]["Authorization"] == f"Bearer {access}"
            assert kwargs["params"]["q"] == "is:unread"
            return Response()

    monkeypatch.setattr("app.capabilities.gmail.httpx.AsyncClient", Client)
    result = await list_gmail_messages("user-one", query="is:unread")
    assert result["messages"] == [{"id": "m1", "threadId": "t1"}]
    with pytest.raises(ValueError, match="not connected"):
        await list_gmail_messages("user-two")


@pytest.mark.asyncio
async def test_gmail_send_requires_explicit_approval(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    upsert_integration_token("user-one", "gmail", cipher.encrypt(b"gmail-secret"), None, "Bearer", "gmail.modify", None)
    with pytest.raises(PermissionError, match="explicit approval"):
        await send_gmail_message("user-one", "to@example.com", "Subject", "Body")


@pytest.mark.asyncio
async def test_gmail_get_message_returns_provider_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    upsert_integration_token("user-one", "gmail", cipher.encrypt(b"gmail-secret"), None, "Bearer", "gmail.modify", None)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"id": "m1", "threadId": "t1", "snippet": "hello"}

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url, **kwargs):
            assert url.endswith("/messages/m1")
            return Response()

    monkeypatch.setattr("app.capabilities.gmail.httpx.AsyncClient", Client)
    result = await get_gmail_message("user-one", "m1")
    assert result["message"]["id"] == "m1"
    assert result["secrets_exposed"] is False
