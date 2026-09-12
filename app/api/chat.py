from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.conversation_store import ConversationStore
from app.ai.file_context import read_text_file
from app.ai.intent import classify_intent
from app.ai.service import generate_reply
from app.ai.tools import run_tool

router = APIRouter(tags=["chat"])
_store = ConversationStore()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    file_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


def _principal(request: Request) -> str:
    return str(getattr(request.state, "principal_id", "")).strip()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    user_id = _principal(request)
    conversation_id = body.conversation_id or str(uuid4())
    try:
        if _store.is_closed(conversation_id, user_id=user_id):
            conversation_id = str(uuid4())
        history = _store.recent(conversation_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    intent = classify_intent(body.message)
    document_context = ""

    if body.file_id:
        try:
            document_context = read_text_file(body.file_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if intent.needs_calculation and not body.file_id:
        expression = body.message
        for marker in ("calculate", "what is", "=", "ಲೆಕ್ಕ"):
            expression = expression.replace(marker, " ")
        tool_result = run_tool("calculator", expression.strip())
        if tool_result.safe:
            reply = tool_result.output
        else:
            reply = user_safe_failure()
    else:
        try:
            reply = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    quality = assess_answer(body.message, reply)
    if not quality.passed:
        reply = user_safe_failure()

    try:
        _store.append(
            conversation_id,
            [("user", body.message), ("assistant", reply)],
            user_id=user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ChatResponse(conversation_id=conversation_id, reply=reply)
