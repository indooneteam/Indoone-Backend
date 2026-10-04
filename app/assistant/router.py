from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.auth import extract_principal
from app.assistant.gemini_live import (
    AssistantGeminiError,
    AssistantGeminiLiveSession,
    translate_upstream_event,
)

logger = logging.getLogger("indoone.assistant.router")
router = APIRouter(tags=["assistant"])

_MAX_CLIENT_MESSAGE_BYTES = 1_000_000


def _authenticate(websocket: WebSocket) -> str:
    authorization = websocket.headers.get("authorization", "")
    if not authorization:
        raise ValueError("bearer authentication required")
    return extract_principal(authorization)


async def _forward_upstream(
    websocket: WebSocket,
    session: AssistantGeminiLiveSession,
) -> None:
    while True:
        payload = await session.receive_event()
        if payload is None:
            return
        for event in translate_upstream_event(payload):
            await websocket.send_json(event)


@router.websocket("/assistant/session")
async def assistant_session(websocket: WebSocket) -> None:
    try:
        user_id = _authenticate(websocket)
    except (RuntimeError, ValueError) as exc:
        await websocket.close(code=1008, reason=str(exc)[:120])
        return

    await websocket.accept()
    session_id = uuid4().hex
    await websocket.send_json(
        {
            "type": "ready",
            "protocol": "indoone.assistant.v1",
            "session_id": session_id,
            "model": "gemini-3.8-live",
        }
    )

    logger.info(
        "Home assistant session opened user_id=%s session_id=%s",
        user_id,
        session_id,
    )

    try:
        async with AssistantGeminiLiveSession() as gemini:
            upstream_task = asyncio.create_task(_forward_upstream(websocket, gemini))
            try:
                while True:
                    raw_message = await websocket.receive_text()
                    if len(raw_message.encode("utf-8")) > _MAX_CLIENT_MESSAGE_BYTES:
                        await websocket.send_json(
                            {"type": "error", "detail": "message is too large"}
                        )
                        continue

                    try:
                        message = json.loads(raw_message)
                    except (TypeError, ValueError):
                        await websocket.send_json(
                            {"type": "error", "detail": "message must be valid JSON"}
                        )
                        continue

                    if not isinstance(message, dict):
                        await websocket.send_json(
                            {"type": "error", "detail": "message must be an object"}
                        )
                        continue

                    message_type = str(message.get("type", "")).strip().lower()

                    if message_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message_type == "audio":
                        await gemini.send_audio(
                            str(message.get("audio_base64", "")),
                            str(message.get("mime_type", "audio/pcm;rate=16000")),
                        )
                    elif message_type == "text":
                        await gemini.send_text(str(message.get("text", "")))
                    elif message_type == "audio_stream_end":
                        await gemini.send_audio_stream_end()
                    elif message_type == "stop":
                        await websocket.send_json(
                            {"type": "stopped", "session_id": session_id}
                        )
                        break
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "unsupported Home assistant message type",
                            }
                        )
            finally:
                upstream_task.cancel()
                try:
                    await upstream_task
                except asyncio.CancelledError:
                    pass
    except WebSocketDisconnect:
        logger.info(
            "Home assistant client disconnected user_id=%s session_id=%s",
            user_id,
            session_id,
        )
    except AssistantGeminiError as exc:
        logger.warning(
            "Home assistant Gemini session failed user_id=%s session_id=%s error=%s",
            user_id,
            session_id,
            exc,
        )
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)[:500]})
        except Exception:
            pass
    except Exception:
        logger.exception(
            "Home assistant session failed user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        try:
            await websocket.send_json(
                {"type": "error", "detail": "Home assistant session failed unexpectedly"}
            )
        except Exception:
            pass
