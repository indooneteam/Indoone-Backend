from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.agent_async import execute_agent_async
from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.conversation_store import ConversationStore
from app.ai.file_context import read_text_file
from app.ai.final_answer import synthesize_tool_answer
from app.ai.intent import classify_intent
from app.ai.service import generate_reply
from app.api.dependencies import current_user_id

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
    return current_user_id(request)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    user_id = _principal(request)
    is_new_conversation = body.conversation_id is None
    conversation_id = body.conversation_id or str(uuid4())

    try:
        if is_new_conversation:
            history: list[tuple[str, str]] = []
        elif _store.is_closed(conversation_id, user_id=user_id):
            conversation_id = str(uuid4())
            history = []
        else:
            history = _store.recent(conversation_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    intent = classify_intent(body.message)
    document_context = ""

    if body.file_id:
        try:
            document_context = read_text_file(body.file_id, user_id=user_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    agent_execution = await execute_agent_async(body.message, user_id=user_id)
    has_agent_activity = bool(agent_execution.steps or agent_execution.blocked_steps)

    if has_agent_activity and not body.file_id:
        if agent_execution.results:
            reply = synthesize_tool_answer(agent_execution.results)
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
