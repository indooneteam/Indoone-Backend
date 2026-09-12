from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.conversation_store import ConversationStore
from app.ai.intent import classify_intent
from app.ai.service import generate_reply
from app.ai.tools import run_tool

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
    if _store.is_closed(conversation_id):
        conversation_id = str(uuid4())

    history = _store.recent(conversation_id)
    intent = classify_intent(request.message)

    if intent.needs_calculation:
        expression = request.message
        for marker in ("calculate", "what is", "=", "ಲೆಕ್ಕ"):
            expression = expression.replace(marker, " ")
        tool_result = run_tool("calculator", expression.strip())
        if tool_result.safe:
            reply = tool_result.output
        else:
            reply = user_safe_failure()
    else:
        try:
            reply = await generate_reply(request.message, history=history)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(request.message, reply)
    if not quality.passed:
        reply = user_safe_failure()

    try:
        _store.append(
            conversation_id,
            [("user", request.message), ("assistant", reply)],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ChatResponse(conversation_id=conversation_id, reply=reply)
