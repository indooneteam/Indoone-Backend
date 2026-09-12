from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.conversation_store import ConversationStore
from app.ai.service import generate_reply

router = APIRouter(tags=["chat"])
_store = ConversationStore()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    history = _store.recent(conversation_id)
    try:
        reply = await generate_reply(request.message, history=history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(request.message, reply)
    if not quality.passed:
        reply = user_safe_failure()

    _store.append(
        conversation_id,
        [("user", request.message), ("assistant", reply)],
    )
    return ChatResponse(conversation_id=conversation_id, reply=reply)
