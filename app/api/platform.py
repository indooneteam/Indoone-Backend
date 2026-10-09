from __future__ import annotations

import asyncio
import base64
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.answer_quality import assess_answer
from app.ai.conversation_store import ConversationStore
from app.ai.file_context import read_text_file, save_text_file
from app.ai.memory import extract_memory_candidates
from app.ai.service import generate_reply
from app.capabilities.control_center import record_control_event, replies_enabled

router = APIRouter(tags=["platform"])
_stream_store = ConversationStore()


class FileRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=3_000_000)


class FileResponse(BaseModel):
    file_id: str
    filename: str
    bytes: int
    text_preview: str


@router.post("/files", response_model=FileResponse)
async def upload_file(request: FileRequest) -> FileResponse:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        result = save_text_file(request.filename, content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        file_id=str(result["file_id"]),
        filename=str(result["filename"]),
        bytes=int(result["bytes"]),
        text_preview=str(result["text"])[:5000],
    )


class StreamChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, max_length=128)
    file_id: str | None = Field(default=None, max_length=128)


@router.post("/chat/stream")
async def stream_chat(body: StreamChatRequest, request: Request) -> StreamingResponse:
    if not replies_enabled("android"):
        record_control_event("android", "reply", "skipped", "/api/platform/chat/stream", 503)
        raise HTTPException(status_code=503, detail="Android AI replies are paused in the Control Center")

    conversation_id = body.conversation_id or str(uuid4())
    if _stream_store.is_closed(conversation_id):
        conversation_id = str(uuid4())

    history = _stream_store.recent(conversation_id)
    document_context = ""
    if body.file_id:
        try:
            document_context = read_text_file(body.file_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        reply = await generate_reply(
            body.message,
            history=history,
            document_context=document_context,
        )
    except RuntimeError as exc:
        record_control_event("android", "reply", "failed", "/api/platform/chat/stream", 503)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(body.message, reply)
    if not quality.passed:
        record_control_event("android", "reply", "failed", "/api/platform/chat/stream", 503)
        raise HTTPException(status_code=503, detail=f"model answer failed quality checks: {quality.reason}")

    try:
        _stream_store.append(
            conversation_id,
            [("user", body.message), ("assistant", reply)],
        )
    except ValueError as exc:
        record_control_event("android", "reply", "failed", "/api/platform/chat/stream", 409)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    record_control_event("android", "reply", "success", "/api/platform/chat/stream", 200)

    async def events():
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id}, ensure_ascii=False)}\\n\\n"
        for chunk_start in range(0, len(reply), 48):
            payload = {"type": "delta", "text": reply[chunk_start : chunk_start + 48]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\\n\\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\\n\\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ai/memory/candidates")
async def memory_candidates(request: PlanRequest) -> dict[str, object]:
    candidates = extract_memory_candidates(request.message)
    return {
        "candidates": [
            {
                "key": item.key,
                "value": item.value,
                "confidence": item.confidence,
                "reason": item.reason,
            }
            for item in candidates
        ]
    }
