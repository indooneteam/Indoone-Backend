from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.conversation_store import ConversationStore
from app.api.dependencies import current_user_id

router = APIRouter(tags=["conversations"])
_store = ConversationStore()


class ConversationSummary(BaseModel):
    conversation_id: str
    status: str
    created_at: str
    updated_at: str
    closed_at: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class ConversationActionResponse(BaseModel):
    conversation_id: str
    status: str


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(request: Request, limit: int = Field(default=50, ge=1, le=100)) -> ConversationListResponse:
    user_id = current_user_id(request)
    return ConversationListResponse(conversations=[ConversationSummary(**item) for item in _store.list_for_user(user_id, limit)])


@router.post("/conversations/{conversation_id}/close", response_model=ConversationActionResponse)
async def close_conversation(request: Request, conversation_id: str) -> ConversationActionResponse:
    user_id = current_user_id(request)
    try:
        _store.close(conversation_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ConversationActionResponse(conversation_id=conversation_id, status="closed")


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(request: Request, conversation_id: str) -> None:
    user_id = current_user_id(request)
    try:
        _store.delete(conversation_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
