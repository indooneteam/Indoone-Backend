from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.conversation_store import ConversationStore
from app.ai.file_context import read_text_file
from app.ai.service import generate_reply
from app.api.dependencies import current_user_id
from app.capabilities.control_center import record_control_event, replies_enabled

router = APIRouter(tags=["chat"])
_store = ConversationStore()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    file_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatSource(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    sources: list[ChatSource] = Field(default_factory=list)


def _principal(request: Request) -> str:
    return current_user_id(request)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    user_id = _principal(request)
    if not replies_enabled("android"):
        record_control_event("android", "reply", "skipped", "/api/chat", 503)
        raise HTTPException(status_code=503, detail="Android AI replies are paused in the Control Center")
    is_new_conversation = body.conversation_id is None
    conversation_id = body.conversation_id or str(uuid4())

    try:
        if is_new_conversation:
            history: list[tuple[str, str]] = []
        elif _store.is_closed(conversation_id, user_id=user_id):
            conversation_id = str(uuid4())
            history = []
        else:
            # Render Free instances have an ephemeral filesystem, so the backend's
            # local SQLite history can disappear after a restart while Android still
            # has the durable Firestore conversation id. A missing local record must
            # not turn an otherwise valid chat request into HTTP 404.
            try:
                history = _store.recent(conversation_id, user_id=user_id)
            except ValueError:
                history = []
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document_context = ""

    if body.file_id:
        try:
            document_context = read_text_file(body.file_id, user_id=user_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        reply = await generate_reply(
            body.message,
            history=history,
            document_context=document_context,
        )
    except RuntimeError as exc:
        record_control_event("android", "reply", "failed", "/api/chat", 503)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
    record_control_event("android", "reply", "success", "/api/chat", 200)
    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        sources=[],
    )
