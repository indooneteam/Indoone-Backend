from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import vibe as vibe_api
from app.main import app


def test_gemini_live_setup_uses_audio_and_transcription() -> None:
    setup = vibe_api._setup_message("models/gemini-live-test", "Aoede")["setup"]

    assert setup["model"] == "models/gemini-live-test"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Aoede"
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}


class FakeGeminiWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []

    async def __aenter__(self) -> "FakeGeminiWebSocket":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if "setup" in message:
            await self.incoming.put(json.dumps({"setupComplete": {}}))
            return

        realtime = message.get("realtimeInput")
        if not isinstance(realtime, dict):
            return
        if isinstance(realtime.get("text"), str):
            await self.incoming.put(json.dumps({
                "serverContent": {
                    "outputTranscription": {"text": "Hello from Gemini Live."},
                },
            }))
        elif isinstance(realtime.get("audio"), dict):
            await self.incoming.put(json.dumps({
                "serverContent": {
                    "inputTranscription": {"text": "Hello"},
                    "outputTranscription": {"text": "Hi there."},
                    "modelTurn": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/pcm;rate=24000",
                                    "data": "AQID",
                                },
                            },
                        ],
                    },
                },
            }))

    async def recv(self) -> str:
        return await self.incoming.get()


@pytest.mark.parametrize("path", ["/api/vibe/session", "/api/live/session"])
def test_vibe_websocket_connects_forwards_text_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-live-test")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "test-user")
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(path, headers={"Authorization": "Bearer test-firebase-token"}) as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            assert ready["protocol"] == "indoone.vibe.v1"

            websocket.send_json({"type": "text", "text": "Hi"})
            transcript = websocket.receive_json()
            assert transcript == {
                "type": "transcript",
                "role": "assistant",
                "text": "Hello from Gemini Live.",
            }

            websocket.send_json({"type": "stop"})
            stopped = websocket.receive_json()
            assert stopped["type"] == "stopped"

    assert upstream.sent[0]["setup"]["model"] == "models/gemini-live-test"
    assert upstream.sent[1]["realtimeInput"] == {"text": "Hi"}


def test_vibe_websocket_forwards_audio_and_translates_audio_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-live-test")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "test-user")
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({
                "type": "audio",
                "audio_base64": "AQID",
                "mime_type": "audio/pcm;rate=16000",
            })

            transcript_user = websocket.receive_json()
            transcript_assistant = websocket.receive_json()
            audio = websocket.receive_json()
            assert transcript_user == {"type": "transcript", "role": "user", "text": "Hello"}
            assert transcript_assistant == {"type": "transcript", "role": "assistant", "text": "Hi there."}
            assert audio == {
                "type": "audio",
                "audio_base64": "AQID",
                "mime_type": "audio/pcm;rate=24000",
                "sample_rate_hz": 24_000,
            }

            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["type"] == "stopped"

    assert upstream.sent[1]["realtimeInput"]["audio"] == {
        "data": "AQID",
        "mimeType": "audio/pcm;rate=16000",
    }


def test_vibe_websocket_reports_missing_live_model_instead_of_silent_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.delenv("GEMINI_LIVE_MODEL", raising=False)
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "test-user")

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            event = websocket.receive_json()
            assert event == {"type": "error", "detail": "GEMINI_LIVE_MODEL is not configured"}


def test_vibe_websocket_rejects_missing_authentication() -> None:
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/vibe/session"):
                pass
