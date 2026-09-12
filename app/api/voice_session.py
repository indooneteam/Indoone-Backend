from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.capabilities.voice_session import (
    VoiceSessionState,
    handle_audio_message,
    handle_speak_message,
    normalize_request_id,
    ready_event,
)

router = APIRouter(tags=["voice"])


@router.websocket("/voice/session")
async def voice_session(websocket: WebSocket) -> None:
    await websocket.accept()
    state = VoiceSessionState()
    await websocket.send_json(ready_event(state))
    try:
        while not state.closed:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await websocket.send_json({"type": "error", "detail": "message must be an object"})
                continue

            message_type = str(message.get("type", "")).strip().lower()
            request_id = normalize_request_id(message.get("request_id"))

            if message_type == "ping":
                await websocket.send_json({"type": "pong", "request_id": request_id, "sequence": state.next_sequence()})
                continue

            if message_type == "stop":
                state.closed = True
                await websocket.send_json({
                    "type": "stopped",
                    "request_id": request_id,
                    "session_id": state.session_id,
                    "sequence": state.next_sequence(),
                })
                continue

            try:
                state.accept_message()
                if message_type == "audio":
                    event = await handle_audio_message(
                        state,
                        str(message.get("audio_base64", "")),
                        mime_type=str(message.get("mime_type", "audio/wav")),
                        language=str(message.get("language", "")),
                        final=bool(message.get("final", True)),
                        request_id=request_id,
                    )
                elif message_type == "speak":
                    event = await handle_speak_message(
                        state,
                        str(message.get("text", "")),
                        language=str(message.get("language", "")),
                        voice=str(message.get("voice", "")),
                        format=str(message.get("format", "wav")),
                        request_id=request_id,
                    )
                else:
                    await websocket.send_json({"type": "error", "request_id": request_id, "detail": "unsupported message type"})
                    continue
                await websocket.send_json(event)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "request_id": request_id, "detail": str(exc)})
            except RuntimeError as exc:
                await websocket.send_json({"type": "error", "request_id": request_id, "detail": str(exc)})
    except WebSocketDisconnect:
        return
