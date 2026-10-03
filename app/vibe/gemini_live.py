from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets

from app.vibe.config import VibeConfig


logger = logging.getLogger("indoone.vibe.gemini")


class VibeGeminiError(RuntimeError):
    """Raised when the Gemini Live upstream session cannot be used safely."""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _server_error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = _text(error.get("message"))
        if message:
            return message[:500]
    return "Gemini Live returned an upstream error"


class GeminiLiveSession:
    def __init__(self, config: VibeConfig | None = None) -> None:
        self.config = config or VibeConfig.from_environment()
        self._socket: Any = None

    async def __aenter__(self) -> "GeminiLiveSession":
        try:
            self._socket = await websockets.connect(
                self.config.websocket_url(),
                open_timeout=15,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            )
            await self._socket.send(json.dumps(self.config.setup_message()))
            raw = await asyncio.wait_for(self._socket.recv(), timeout=15)
        except Exception as exc:
            await self._close_socket()
            logger.warning("failed to open Gemini Live session: %s", exc)
            raise VibeGeminiError("Vibe upstream session could not be opened") from exc

        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            await self._close_socket()
            raise VibeGeminiError("Vibe upstream returned invalid setup data") from exc

        if not isinstance(payload, dict):
            await self._close_socket()
            raise VibeGeminiError("Vibe upstream returned invalid setup data")

        if "error" in payload:
            message = _server_error_message(payload)
            await self._close_socket()
            raise VibeGeminiError(message)

        if "setupComplete" not in payload:
            await self._close_socket()
            raise VibeGeminiError("Vibe upstream session was not ready")

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._close_socket()

    async def _close_socket(self) -> None:
        socket = self._socket
        self._socket = None
        if socket is None:
            return
        try:
            await socket.close()
        except Exception:
            logger.debug("Gemini Live socket close failed", exc_info=True)

    def _require_socket(self) -> Any:
        if self._socket is None:
            raise VibeGeminiError("Vibe upstream session is not connected")
        return self._socket

    async def send_audio(self, audio_base64: str, mime_type: str) -> None:
        clean_audio = _text(audio_base64)
        if not clean_audio:
            raise ValueError("audio_base64 is required")

        clean_mime = _text(mime_type) or "audio/pcm;rate=16000"
        if clean_mime != "audio/pcm;rate=16000":
            raise ValueError("Vibe audio must use raw PCM 16kHz input")

        message = {
            "realtimeInput": {
                "audio": {
                    "data": clean_audio,
                    "mimeType": clean_mime,
                }
            }
        }
        try:
            await self._require_socket().send(json.dumps(message))
        except Exception as exc:
            raise VibeGeminiError("failed to send Vibe audio upstream") from exc

    async def send_text(self, text: str) -> None:
        clean_text = _text(text)
        if not clean_text:
            raise ValueError("text is required")
        if len(clean_text) > 8_000:
            raise ValueError("text is too long")

        message = {"realtimeInput": {"text": clean_text}}
        try:
            await self._require_socket().send(json.dumps(message))
        except Exception as exc:
            raise VibeGeminiError("failed to send Vibe text upstream") from exc

    async def send_audio_stream_end(self) -> None:
        try:
            await self._require_socket().send(
                json.dumps({"realtimeInput": {"audioStreamEnd": True}})
            )
        except Exception as exc:
            raise VibeGeminiError("failed to end Vibe audio stream") from exc

    async def receive_event(self) -> dict[str, Any] | None:
        try:
            raw = await self._require_socket().recv()
        except Exception as exc:
            if self._socket is None:
                return None
            raise VibeGeminiError("Vibe upstream connection was closed") from exc

        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise VibeGeminiError("Vibe upstream returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise VibeGeminiError("Vibe upstream returned an invalid event")

        if "error" in payload:
            raise VibeGeminiError(_server_error_message(payload))

        return payload


def translate_upstream_event(payload: dict[str, Any]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    if "setupComplete" in payload:
        events.append({"type": "connected"})
        return events

    server_content = payload.get("serverContent")
    if isinstance(server_content, dict):
        interaction_status = _text(server_content.get("interactionStatus"))
        if interaction_status:
            events.append(
                {
                    "type": "status",
                    "interaction_status": interaction_status,
                }
            )

        interrupted = server_content.get("interrupted")
        if interrupted is True:
            events.append({"type": "interrupted"})

        interim_input = server_content.get("interimInputTranscription")
        if isinstance(interim_input, dict):
            transcript = _text(interim_input.get("text"))
            if transcript:
                events.append(
                    {
                        "type": "transcript",
                        "role": "user",
                        "final": False,
                        "text": transcript,
                    }
                )

        input_transcription = server_content.get("inputTranscription")
        if isinstance(input_transcription, dict):
            transcript = _text(input_transcription.get("text"))
            if transcript:
                events.append(
                    {
                        "type": "transcript",
                        "role": "user",
                        "final": True,
                        "text": transcript,
                    }
                )

        output_transcription = server_content.get("outputTranscription")
        if isinstance(output_transcription, dict):
            transcript = _text(output_transcription.get("text"))
            if transcript:
                events.append(
                    {
                        "type": "transcript",
                        "role": "assistant",
                        "final": True,
                        "text": transcript,
                    }
                )

        model_turn = server_content.get("modelTurn")
        if isinstance(model_turn, dict):
            parts = model_turn.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline_data = part.get("inlineData")
                    if not isinstance(inline_data, dict):
                        continue
                    audio_data = _text(inline_data.get("data"))
                    mime_type = _text(inline_data.get("mimeType"))
                    if audio_data:
                        events.append(
                            {
                                "type": "audio",
                                "audio_base64": audio_data,
                                "mime_type": mime_type or "audio/pcm;rate=24000",
                            }
                        )

        if server_content.get("turnComplete") is True:
            events.append(
                {
                    "type": "turn_complete",
                    "interaction_status": interaction_status or "IDLE",
                }
            )

        waiting_for_input = server_content.get("waitingForInput")
        if waiting_for_input is True:
            events.append({"type": "waiting_for_input"})

    if "toolCall" in payload:
        events.append({"type": "tool_call", "payload": payload["toolCall"]})

    if not events:
        events.append({"type": "upstream_event", "payload": payload})

    return events
