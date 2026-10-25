from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.quality.answer_quality import assess_answer
from app.ai.execution.tool_loop import run_tool_loop
from app.ai.memory.conversation_store import ConversationStore
from app.ai.memory.file_context import read_text_file
from app.ai.memory.memory import extract_memory_candidates
from app.ai.gateway.service import generate_reply
from app.api.errors import REQUEST_FAILED_MESSAGE, provider_failure

router = APIRouter(tags=["platform"])
_stream_store = ConversationStore()


class StreamChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, max_length=128)
    file_id: str | None = Field(default=None, max_length=128)


class MemoryCandidatesRequest(BaseModel):
    """The message to read durable memory candidates from."""

    message: str = Field(min_length=1, max_length=20_000)


# The Control Center gateway expects /api/platform/chat/stream; both paths map
# to this one handler.
@router.post("/platform/chat/stream")
@router.post("/chat/stream")
async def stream_chat(request: Request, body: StreamChatRequest) -> StreamingResponse:
    user_id = str(getattr(request.state, "principal_id", "")).strip()
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
        # Use the same unified tool loop as /api/chat so streaming answers get
        # the user's tools too. It only runs for a signed-in user.
        tool_reply: str | None = None
        if user_id:
            try:
                tool_reply = await run_tool_loop(user_id, conversation_id, body.message, history=history)
            except Exception:
                tool_reply = None
        reply = tool_reply if tool_reply is not None else await generate_reply(
            body.message,
            history=history,
            document_context=document_context,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(body.message, reply)
    if not quality.passed:
        logger.error("Model answer failed quality checks: %s", quality.reason)
        raise HTTPException(status_code=503, detail=REQUEST_FAILED_MESSAGE)

    try:
        _stream_store.append(
            conversation_id,
            [("user", body.message), ("assistant", reply)],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def events():
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
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


@router.post("/ai/memory/candidates")
async def memory_candidates(request: MemoryCandidatesRequest) -> dict[str, object]:
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
