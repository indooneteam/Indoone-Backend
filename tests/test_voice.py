from __future__ import annotations

import asyncio
import base64

import httpx
from fastapi.testclient import TestClient

from app.capabilities.voice import synthesize_speech, transcribe_audio
from app.main import app


class MockTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "stt.local":
            return httpx.Response(200, json={"text": "namaskara", "language": "kn", "provider": "local", "model": "stt-test", "confidence": 0.91})
        if request.url.host == "tts.local":
            audio = base64.b64encode(b"fake-wav").decode("ascii")
            return httpx.Response(200, json={"audio_base64": audio, "mime_type": "audio/wav", "provider": "local", "model": "tts-test", "sample_rate_hz": 16_000})
        return httpx.Response(404)


_original_async_client = httpx.AsyncClient


def client_factory(*args, **kwargs):
    kwargs["transport"] = MockTransport()
    return _original_async_client(*args, **kwargs)


def test_transcribe_audio_normalizes_local_result(monkeypatch) -> None:
    monkeypatch.setattr("app.capabilities.voice.httpx.AsyncClient", client_factory)
    monkeypatch.setenv("INDOONE_STT_URL", "http://stt.local/transcribe")

    encoded = base64.b64encode(b"fake-audio").decode("ascii")
    result = asyncio.run(transcribe_audio(encoded, mime_type="audio/wav", language="kn"))

    assert result.text == "namaskara"
    assert result.language == "kn"
    assert result.model == "stt-test"
    assert result.confidence == 0.91


def test_synthesize_speech_normalizes_local_result(monkeypatch) -> None:
    monkeypatch.setattr("app.capabilities.voice.httpx.AsyncClient", client_factory)
    monkeypatch.setenv("INDOONE_TTS_URL", "http://tts.local/synthesize")

    result = asyncio.run(synthesize_speech("namaskara", language="kn"))

    assert result.audio_base64 == base64.b64encode(b"fake-wav").decode("ascii")
    assert result.mime_type == "audio/wav"
    assert result.sample_rate_hz == 16_000


def test_voice_endpoints_return_503_without_runtime(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_STT_URL", raising=False)
    monkeypatch.delenv("INDOONE_TTS_URL", raising=False)
    with TestClient(app) as client:
        encoded = base64.b64encode(b"fake-audio").decode("ascii")
        stt = client.post("/api/voice/transcribe", json={"audio_base64": encoded})
        tts = client.post("/api/voice/synthesize", json={"text": "hello"})
    assert stt.status_code == 503
    assert tts.status_code == 503


def test_voice_websocket_protocol(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_STT_URL", raising=False)
    monkeypatch.delenv("INDOONE_TTS_URL", raising=False)
    with TestClient(app) as client:
        with client.websocket_connect("/api/voice/session") as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            assert ready["protocol"] == "indoone.voice.v1"
            assert ready["session_id"]
            assert ready["capabilities"] == ["stt", "tts", "partial_transcripts", "request_ids", "cancellation"]

            websocket.send_json({"type": "ping", "request_id": "ping-1"})
            assert websocket.receive_json() == {"type": "pong", "request_id": "ping-1", "sequence": 2}

            websocket.send_json({"type": "stop", "request_id": "stop-1"})
            stopped = websocket.receive_json()
            assert stopped["type"] == "stopped"
            assert stopped["request_id"] == "stop-1"
            assert stopped["session_id"] == ready["session_id"]


def test_voice_websocket_audio_and_tts(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_STT_URL", "http://stt.local/transcribe")
    monkeypatch.setenv("INDOONE_TTS_URL", "http://tts.local/synthesize")
    monkeypatch.setattr("app.capabilities.voice.httpx.AsyncClient", client_factory)

    encoded = base64.b64encode(b"fake-audio").decode("ascii")
    with TestClient(app) as client:
        with client.websocket_connect("/api/voice/session") as websocket:
            ready = websocket.receive_json()
            websocket.send_json({"type": "audio", "audio_base64": encoded, "language": "kn", "final": False, "request_id": "stt-1"})
            transcript = websocket.receive_json()
            assert transcript["type"] == "transcript"
            assert transcript["final"] is False
            assert transcript["request_id"] == "stt-1"
            assert transcript["text"] == "namaskara"
            assert transcript["language"] == "kn"
            assert transcript["sequence"] == 2

            websocket.send_json({"type": "speak", "text": "namaskara", "language": "kn", "request_id": "tts-1"})
            audio = websocket.receive_json()
            assert audio["type"] == "audio"
            assert audio["request_id"] == "tts-1"
            assert audio["audio_base64"] == base64.b64encode(b"fake-wav").decode("ascii")
            assert audio["mime_type"] == "audio/wav"
            assert audio["sample_rate_hz"] == 16_000
            assert audio["sequence"] == 3

            assert ready["session_id"]
