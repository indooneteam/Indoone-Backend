import os

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.ai.memory.conversation_store import ConversationStore
from app.api.security.dependencies import current_user_id

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


class RenameThreadRequest(BaseModel):
    title: str


class ShareLinkResponse(BaseModel):
    share_url: str


class SharedMessage(BaseModel):
    role: str
    text: str


class SharedThreadResponse(BaseModel):
    messages: list[SharedMessage]


class ConversationActionResponse(BaseModel):
    conversation_id: str
    status: str


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    request: Request, limit: int = Query(default=50, ge=1, le=100)
) -> ConversationListResponse:
    user_id = current_user_id(request)
    return ConversationListResponse(
        conversations=[ConversationSummary(**item) for item in _store.list_for_user(user_id, limit)]
    )


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


def _share_base_url() -> str:
    """Where a share link points. Overridable so staging does not mint production links."""
    base = os.getenv("INDOONE_SHARE_BASE_URL", "").strip() or "https://indoone.ai/share"

    return base.rstrip("/")


@router.post("/chat/threads/{thread_id}/share", response_model=ShareLinkResponse)
async def share_thread(request: Request, thread_id: str) -> ShareLinkResponse:
    """Mint a read-only link to this thread.

    The thread is not copied. The link reads the conversation live, so it cannot drift from
    what the owner believes they shared - and deleting the row is how sharing stops.
    """
    user_id = current_user_id(request)

    try:
        share_id = _store.create_share(thread_id, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ShareLinkResponse(share_url=f"{_share_base_url()}/{share_id}")


@router.patch("/chat/threads/{thread_id}", response_model=ConversationActionResponse)
async def rename_thread(
    request: Request, thread_id: str, body: RenameThreadRequest
) -> ConversationActionResponse:
    user_id = current_user_id(request)

    try:
        title = _store.rename(thread_id, body.title, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ConversationActionResponse(conversation_id=thread_id, status=title)


@router.get("/chat/shared/{share_id}", response_model=SharedThreadResponse)
async def read_shared_thread(share_id: str) -> SharedThreadResponse:
    """Public and read-only: this is what the link is for.

    Deliberately unauthenticated, and deliberately unable to write. It returns the messages
    and nothing about the owner - no user id, no conversation id, no status.
    """
    messages = _store.shared_thread(share_id)

    if messages is None:
        raise HTTPException(status_code=404, detail="This share link does not exist")

    return SharedThreadResponse(
        messages=[SharedMessage(role=role, text=text) for role, text in messages]
    )
