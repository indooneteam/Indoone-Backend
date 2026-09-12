from __future__ import annotations

import asyncio
import base64
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.file_context import save_text_file
from app.ai.service import generate_reply

router = APIRouter(tags=["platform"])


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


@router.post("/chat/stream")
async def stream_chat(request: StreamChatRequest) -> StreamingResponse:
    try:
        reply = await generate_reply(request.message, history=[])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(request.message, reply)
    if not quality.passed:
        reply = user_safe_failure()

    async def events():
        for chunk_start in range(0, len(reply), 48):
            payload = {"type": "delta", "text": reply[chunk_start : chunk_start + 48]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
