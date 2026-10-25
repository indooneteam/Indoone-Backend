from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.voice import vibe as vibe_api
from app.main import app


def test_gemini_live_setup_uses_audio_and_transcription() -> None:
    setup = vibe_api._setup_message("models/gemini-live-test", "Aoede")["setup"]

    assert setup["model"] == "models/gemini-live-test"
    assert ".v1alpha.GenerativeService.BidiGenerateContent" in vibe_api._GEMINI_LIVE_ENDPOINT
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    # A live voice session must not go quiet while the model reasons, so the default is
    # low rather than medium. See the note on _DEFAULT_THINKING_LEVEL.
    assert setup["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert "functionDeclarations" in setup["tools"][0]
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
                            "name": "github_list_repositories",
                            "args": {},
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
            asyncio.Lock(),
        )

    assert asyncio.run(run()) is True
    assert frontend.events == [{
        "type": "transcript",
        "role": "assistant",
        "text": "I found one unread email.",
    }]


def test_vibe_websocket_rejects_missing_authentication(monkeypatch) -> None:
    """A socket that never authenticates gets no session.

    The socket is now accepted before it is authenticated, because a browser
    cannot put a token in a WebSocket header and sends it as its first message
    instead. So the refusal comes after the short auth window rather than at
    once - but an unauthenticated socket is still closed, and still gets no
    session. The window is shortened here so the test does not wait it out.
    """
    monkeypatch.setattr(vibe_api, "_LIVE_AUTH_TIMEOUT_SECONDS", 0.2)

    with TestClient(app) as client:
        # The socket is accepted first now, so the refusal shows up as the
        # socket closing with nothing to read - never as a session starting.
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/vibe/session") as socket:
                socket.receive_json()



def test_vibe_identity_uses_product_name_without_forcing_a_fixed_phrase() -> None:
    setup = vibe_api._setup_message("models/gemini-live-test", "Aoede")["setup"]
    instruction = setup["systemInstruction"]["parts"][0]["text"]

    assert "Indoone Vibe" in instruction
    assert "Do not call the live voice experience only 'Indoone AI'." in instruction
    assert "Do not force the fixed phrase 'I am Indoone Vibe' as a standalone reply" in instruction
    assert "live voice experience in the Indoone app" in instruction


def _instruction(setup: dict[str, object]) -> str:
    return setup["systemInstruction"]["parts"][0]["text"]


def test_vibe_setup_tells_the_model_which_read_only_tools_it_has() -> None:
    setup = vibe_api._setup_message("models/gemini-live-test", "Aoede")["setup"]
    instruction = _instruction(setup)

    assert "Zoho Mail" in instruction
    assert "email tools" in instruction
    assert "GitHub" in instruction
    assert "read-only tools" in instruction
    assert "Never claim you sent, changed, archived, or deleted anything." in instruction
    # The Gmail-only shortcut must not be advertised to a Zoho Mail user.
    assert "search_gmail_messages" not in instruction


def test_vibe_setup_omits_the_context_lines_when_there_is_no_context() -> None:
    setup = vibe_api._setup_message("models/gemini-live-test", "Aoede")["setup"]
    instruction = _instruction(setup)

    assert "The user's name is" not in instruction
    assert "Today is " not in instruction
    assert instruction == f"{vibe_api._SYSTEM_INSTRUCTION} {vibe_api._TOOL_INSTRUCTION}"


def test_vibe_setup_appends_the_session_context_lines() -> None:
    setup = vibe_api._setup_message(
        "models/gemini-live-test",
        "Aoede",
        "medium",
        "  The user's name is Yashwanth.   Today is 2026-10-13 (Tuesday), 18:10 IST.  ",
    )["setup"]
    instruction = _instruction(setup)

    assert instruction.endswith(
        "The user's name is Yashwanth. Today is 2026-10-13 (Tuesday), 18:10 IST."
    )
    assert "  " not in instruction


def test_vibe_websocket_sends_the_user_name_and_today_in_the_live_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "firebase-user-1")
    monkeypatch.setattr(
        vibe_api,
        "_user_name_line",
        lambda user_id: "The user's name is Yashwanth." if user_id == "firebase-user-1" else "",
    )
    monkeypatch.setattr(
        vibe_api,
        "_now_line",
        lambda: "Today is 2026-10-13 (Tuesday), 18:10 IST.",
    )
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["type"] == "stopped"

    instruction = _instruction(upstream.sent[0]["setup"])
    assert "The user's name is Yashwanth." in instruction
    assert "Today is 2026-10-13 (Tuesday), 18:10 IST." in instruction


def test_vibe_tool_failure_answers_the_model_and_keeps_the_session_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "firebase-user-1")

    async def failing_tool(user_id: str, name: str, args: object) -> dict[str, object]:
        raise RuntimeError("Zoho Mail is not reachable")

    monkeypatch.setattr(vibe_api, "_execute_voice_tool", failing_tool)
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "text", "text": "Check my unread email"})

            assert websocket.receive_json() == {
                "type": "status",
                "interaction_status": "IN_PROGRESS",
            }
            assert websocket.receive_json() == {"type": "status", "interaction_status": "IDLE"}
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
    assert function_response["name"] == "github_list_repositories"
    assert function_response["response"]["output"] == {
        "ok": False,
        "error": "That action could not be completed right now.",
    }


def test_vibe_sends_the_github_summary_instead_of_the_raw_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "ok": True,
        "result": {
            "integration": "github",
            "repositories": [
                {
                    "id": 1,
                    "name": "Indoone-Backend",
                    "full_name": "indooneteam/Indoone-Backend",
                    "private": True,
                    "html_url": "https://github.com/indooneteam/Indoone-Backend",
                    "default_branch": "main",
                    "description": None,
                },
            ],
        },
    }

    async def fake_dispatch(user_id: str, conversation_id: str, name: str, arguments: object):
        assert user_id == "firebase-user-1"
        assert name == "github_list_repositories"
        return raw

    monkeypatch.setattr(vibe_api, "dispatch_tool", fake_dispatch)

    result = asyncio.run(
        vibe_api._execute_voice_tool("firebase-user-1", "github_list_repositories", {})
    )

    assert result["ok"] is True
    assert "indooneteam/Indoone-Backend" in result["output"]
    assert "Private" in result["output"]
    assert "repositories" not in result


def test_vibe_reports_a_github_failure_as_readable_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_dispatch(user_id: str, conversation_id: str, name: str, arguments: object):
        return {"ok": False, "code": "GITHUB_TOOL_FAILED", "message": "GitHub is not connected."}

    monkeypatch.setattr(vibe_api, "dispatch_tool", fake_dispatch)

    result = asyncio.run(
        vibe_api._execute_voice_tool("firebase-user-1", "github_list_repositories", {})
    )

    assert result == {"ok": False, "error": "GitHub is not connected."}


def test_vibe_sends_its_own_message_when_gemini_reports_an_error() -> None:
    class FakeFrontend:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    frontend = FakeFrontend()

    async def run() -> bool:
        return await vibe_api._handle_gemini_message(
            frontend,  # type: ignore[arg-type]
            json.dumps({"error": {"code": 13, "message": "A system error occurred"}}),
            "test-api-key",
            "user-1",
            object(),
            asyncio.Lock(),
            set(),
            asyncio.Lock(),
        )

    assert asyncio.run(run()) is False
    assert frontend.events == [
        {"type": "error", "detail": vibe_api._VOICE_ERROR_MESSAGE}
    ]
    assert "system error" not in json.dumps(frontend.events)


def test_vibe_handles_a_cancelled_tool_call_without_crashing() -> None:
    class FakeFrontend:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    frontend = FakeFrontend()

    async def run() -> bool:
        return await vibe_api._handle_gemini_message(
            frontend,  # type: ignore[arg-type]
            json.dumps({"toolCallCancellation": {"ids": ["call-1"]}}),
            "test-api-key",
            "user-1",
            object(),
            asyncio.Lock(),
            set(),
            asyncio.Lock(),
        )

    assert asyncio.run(run()) is True
    assert frontend.events == []


def _pretend_connected(monkeypatch: pytest.MonkeyPatch, connected: set[str]) -> None:
    """Make both the chat catalogue and Vibe see only these connectors."""
    from app.ai.execution import tool_loop as tool_loop_api
    def lookup(user_id: str, connector_id: str):
        return {"integration_id": connector_id} if connector_id in connected else None

    monkeypatch.setattr(tool_loop_api, "get_integration_token_metadata", lookup)
    monkeypatch.setattr(vibe_api, "get_integration_token_metadata", lookup)


def _voice_tool_names(monkeypatch: pytest.MonkeyPatch, connected: set[str], user: str = "firebase-user-1") -> list[str]:
    _pretend_connected(monkeypatch, connected)
    return [str(item.get("name")) for item in vibe_api._voice_tool_declarations(user)]


def test_vibe_offers_only_the_connected_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    names = _voice_tool_names(monkeypatch, {"github", "zoho_mail"})

    assert "github_list_repositories" in names
    assert "search_emails" in names
    assert "search_gmail_messages" not in names
    for prefix in ("google_", "youtube_", "instagram_", "facebook_", "canva_"):
        assert not [name for name in names if name.startswith(prefix)]


def test_vibe_hides_every_connector_tool_when_nothing_is_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _voice_tool_names(monkeypatch, set()) == []


def test_vibe_keeps_the_full_catalogue_without_a_signed_in_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = _voice_tool_names(monkeypatch, set(), user="")

    assert "github_list_repositories" in names
    assert "github_list_repositories" in names
    assert "instagram_list_media" in names


def test_vibe_live_setup_uses_the_signed_in_users_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
    monkeypatch.setattr(vibe_api, "extract_principal", lambda authorization: "firebase-user-1")
    _pretend_connected(monkeypatch, {"github"})
    upstream = FakeGeminiWebSocket()
    monkeypatch.setattr(vibe_api.websockets, "connect", lambda *args, **kwargs: upstream)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/vibe/session",
            headers={"Authorization": "Bearer test-firebase-token"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["type"] == "stopped"

    declarations = upstream.sent[0]["setup"]["tools"][0]["functionDeclarations"]
    names = [str(item.get("name")) for item in declarations]
    assert "github_list_repositories" in names
    assert "google_calendar_list_events" not in names
    assert "canva_list_designs" not in names
