import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.api.notifications as notifications_api


OWNER = "user-one"


def _request(user_id: str = OWNER):
    return SimpleNamespace(state=SimpleNamespace(principal_id=user_id))


def test_register_notification_token_requires_auth() -> None:
    body = notifications_api.RegisterTokenRequest(token="device-token-1")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(notifications_api.register_notification_token(_request(""), body))

    assert exc.value.status_code == 401


def test_register_notification_token_rejects_blank_token() -> None:
    body = notifications_api.RegisterTokenRequest(token="   ")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(notifications_api.register_notification_token(_request(), body))

    assert exc.value.status_code == 422


def test_register_notification_token_stores_for_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("INDOONE_NOTIFICATION_DB", str(tmp_path / "notifications.db"))
    notifications_api.initialize()

    body = notifications_api.RegisterTokenRequest(token="device-token-1")
    result = asyncio.run(notifications_api.register_notification_token(_request(), body))

    assert result == {"ok": True, "registered": True, "platform": "android"}
    assert notifications_api.stored_tokens(OWNER) == ["device-token-1"]


def test_register_notification_token_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("INDOONE_NOTIFICATION_DB", str(tmp_path / "notifications.db"))
    notifications_api.initialize()

    body = notifications_api.RegisterTokenRequest(token="device-token-1")
    asyncio.run(notifications_api.register_notification_token(_request(), body))
    asyncio.run(notifications_api.register_notification_token(_request(), body))

    assert notifications_api.stored_tokens(OWNER) == ["device-token-1"]


def test_stored_tokens_are_scoped_to_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("INDOONE_NOTIFICATION_DB", str(tmp_path / "notifications.db"))
    notifications_api.initialize()

    first = notifications_api.RegisterTokenRequest(token="device-token-1")
    second = notifications_api.RegisterTokenRequest(token="device-token-2")
    asyncio.run(notifications_api.register_notification_token(_request("user-one"), first))
    asyncio.run(notifications_api.register_notification_token(_request("user-two"), second))

    assert notifications_api.stored_tokens("user-one") == ["device-token-1"]
    assert notifications_api.stored_tokens("user-two") == ["device-token-2"]
