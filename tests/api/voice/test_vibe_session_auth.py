"""The live Vibe session must accept the token both ways.

The native app sends it in the Authorization header. A browser cannot set a
header on a WebSocket at all, so the web app sends it as the opening message.
"""
import asyncio

import pytest

from app.api.voice import vibe as vibe_module
class FakeSocket:
    """Stands in for a WebSocket, recording what it was asked for."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = None

    async def receive_json(self):
        if not self._messages:
            await asyncio.sleep(5)
            raise AssertionError("the test gave no more messages")
        item = self._messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)


def _token(monkeypatch, user="user-one"):
    import base64
    import hashlib
    import hmac
    import time

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    encode = lambda b: base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")
    payload = f"{encode(user.encode())}.{encode(str(int(time.time())).encode())}".encode()
    return f"{payload.decode()}.{encode(hmac.new(b'x' * 32, payload, hashlib.sha256).digest())}"


def test_the_first_message_may_carry_the_token(monkeypatch) -> None:
    token = _token(monkeypatch)
    socket = FakeSocket([{"type": "auth", "token": token}])

    assert asyncio.run(vibe_module._await_websocket_auth(socket)) == "user-one"


def test_a_first_message_that_is_not_auth_is_refused(monkeypatch) -> None:
    _token(monkeypatch)
    socket = FakeSocket([{"type": "audio", "audio_base64": "AAAA"}])

    with pytest.raises(ValueError, match="first message must be an auth message"):
        asyncio.run(vibe_module._await_websocket_auth(socket))


def test_an_auth_message_without_a_token_is_refused(monkeypatch) -> None:
    _token(monkeypatch)
    socket = FakeSocket([{"type": "auth", "token": "  "}])

    with pytest.raises(ValueError, match="carried no token"):
        asyncio.run(vibe_module._await_websocket_auth(socket))


def test_a_bad_token_is_refused(monkeypatch) -> None:
    _token(monkeypatch)
    socket = FakeSocket([{"type": "auth", "token": "not-a-real-token"}])

    with pytest.raises(Exception):
        asyncio.run(vibe_module._await_websocket_auth(socket))


def test_a_silent_socket_times_out(monkeypatch) -> None:
    _token(monkeypatch)
    monkeypatch.setattr(vibe_module, "_LIVE_AUTH_TIMEOUT_SECONDS", 0.15)
    socket = FakeSocket([])

    with pytest.raises(Exception):
        asyncio.run(vibe_module._await_websocket_auth(socket))


def test_the_endpoint_closes_a_socket_that_never_authenticates(monkeypatch) -> None:
    """No header, and no auth message: the socket must not be left open."""
    from fastapi.testclient import TestClient

    _token(monkeypatch)
    monkeypatch.setattr(vibe_module, "_LIVE_AUTH_TIMEOUT_SECONDS", 0.2)

    from app.main import app

    client = TestClient(app)

    with pytest.raises(Exception):
        with client.websocket_connect("/api/vibe/session") as socket:
            socket.receive_json()


def test_the_endpoint_reads_the_authorization_header(monkeypatch) -> None:
    """The native app's path: a bad header must be refused, not accepted."""
    from fastapi.testclient import TestClient

    _token(monkeypatch)
    monkeypatch.setattr(vibe_module, "_LIVE_AUTH_TIMEOUT_SECONDS", 0.2)

    from app.main import app

    client = TestClient(app)

    with pytest.raises(Exception):
        with client.websocket_connect(
            "/api/vibe/session", headers={"Authorization": "Bearer nonsense"}
        ) as socket:
            socket.receive_json()
