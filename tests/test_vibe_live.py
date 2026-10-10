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
    assert ".v1alpha.GenerativeService.BidiGenerateContent" in vibe_api._GEMINI_LIVE_ENDPOINT
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}
    tool = setup["tools"][0]["functionDeclarations"][0]
    assert tool["name"] == "search_gmail_messages"
    assert tool["behavior"] == "NON_BLOCKING"
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

        if "toolResponse" in message:
            await self.incoming.put(json.dumps({
                "interactionStatus": "IDLE",
                "serverContent": {
                    "outputTranscription": {"text": "I found one unread email."},
                },
            }))
            return

        realtime = message.get("realtimeInput")
        if not isinstance(realtime, dict):
            return
        if isinstance(realtime.get("text"), str):
            if "email" in realtime["text"].lower():
                await self.incoming.put(json.dumps({
                    "interactionStatus": "IN_PROGRESS",
                    "toolCall": {
                        "functionCalls": [{
                            "id": "call-gmail-1",
                            "name": "search_gmail_messages",
                            "args": {"query": "is:unread", "max_results": 2},
                        }],
                    },
                }))
            else:
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
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
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

    assert upstream.sent[0]["setup"]["model"] == "models/gemini-3.8-live-extended-thinking"
    assert upstream.sent[1]["realtimeInput"] == {"text": "Hi"}


def test_vibe_websocket_forwards_audio_and_translates_audio_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
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


def test_vibe_websocket_rejects_legacy_live_model_for_extended_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "test-user")

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            event = websocket.receive_json()
            assert event == {
                "type": "error",
                "detail": "Indoone Vibe requires GEMINI_LIVE_MODEL=gemini-3.8-live-extended-thinking",
            }


def test_search_gmail_for_voice_uses_user_scoped_read_only_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list(user_id: str, *, query: str, max_results: int):
        assert user_id == "firebase-user-1"
        assert query == "is:unread"
        assert max_results == 3
        return {"messages": [{"id": "msg-1", "threadId": "thread-1"}]}

    async def fake_get(user_id: str, message_id: str):
        assert user_id == "firebase-user-1"
        assert message_id == "msg-1"
        return {
            "message": {
                "id": message_id,
                "threadId": "thread-1",
                "snippet": "Please review the updated estimate.",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "client@example.com"},
                        {"name": "Subject", "value": "Updated estimate"},
                        {"name": "Date", "value": "Sat, 10 Oct 2026 10:00:00 +0000"},
                    ],
                },
            },
        }

    monkeypatch.setattr(vibe_api, "list_gmail_messages", fake_list)
    monkeypatch.setattr(vibe_api, "get_gmail_message", fake_get)

    result = asyncio.run(
        vibe_api._execute_voice_tool(
            "firebase-user-1",
            "search_gmail_messages",
            {"query": "is:unread", "max_results": 3},
        )
    )
    assert result == {
        "ok": True,
        "query": "is:unread",
        "count": 1,
        "emails": [{
            "message_id": "msg-1",
            "thread_id": "thread-1",
            "date": "Sat, 10 Oct 2026 10:00:00 +0000",
            "from": "client@example.com",
            "to": "",
            "subject": "Updated estimate",
            "snippet": "Please review the updated estimate.",
        }],
    }



def test_vibe_executes_nonblocking_gmail_tool_and_tracks_interaction_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "firebase-user-1")

    async def fake_tool(user_id: str, name: str, args: object) -> dict[str, object]:
        assert user_id == "firebase-user-1"
        assert name == "search_gmail_messages"
        assert args == {"query": "is:unread", "max_results": 2}
        await asyncio.sleep(0)
        return {"ok": True, "count": 1, "emails": [{"subject": "Invoice"}]}

    monkeypatch.setattr(vibe_api, "_execute_voice_tool", fake_tool)
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "text", "text": "Check my unread email"})

            in_progress = websocket.receive_json()
            assert in_progress == {"type": "status", "interaction_status": "IN_PROGRESS"}
            idle = websocket.receive_json()
            assert idle == {"type": "status", "interaction_status": "IDLE"}
            transcript = websocket.receive_json()
            assert transcript == {
                "type": "transcript",
                "role": "assistant",
                "text": "I found one unread email.",
            }

            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["type"] == "stopped"

    tool_response = next(message for message in upstream.sent if "toolResponse" in message)
    function_response = tool_response["toolResponse"]["functionResponses"][0]
    assert function_response["id"] == "call-gmail-1"
    assert function_response["name"] == "search_gmail_messages"
    assert function_response["response"]["output"] == {
        "ok": True,
        "count": 1,
        "emails": [{"subject": "Invoice"}],
    }



def test_vibe_does_not_forward_gemini_parts_marked_as_internal_thought() -> None:
    class FakeFrontend:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    frontend = FakeFrontend()

    async def run() -> bool:
        return await vibe_api._handle_gemini_message(
            frontend,  # type: ignore[arg-type]
            json.dumps({
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {"text": "Internal reasoning that must stay hidden.", "thought": True},
                            {"text": "I found one unread email."},
                        ],
                    },
                },
            }),
            "test-api-key",
            "user-1",
            object(),
            asyncio.Lock(),
            set(),
        )

    assert asyncio.run(run()) is True
    assert frontend.events == [{
        "type": "transcript",
        "role": "assistant",
        "text": "I found one unread email.",
    }]


def test_vibe_websocket_rejects_missing_authentication() -> None:
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/vibe/session"):
                pass
