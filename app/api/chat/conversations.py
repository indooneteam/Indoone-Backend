import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ai.memory.conversation_store import ConversationStore
from app.api.chat.share_page import SHARE_PAGE_HEADERS, render_not_found, render_share_page
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
    title: str = ""


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


# A page, not an API: registered without the /api prefix in main.py, because the link
# people paste into a chat is https://indoone.ai/share/<id>.
page_router = APIRouter(tags=["share"])


@page_router.get("/share/{share_id}", response_class=HTMLResponse)
async def shared_thread_page(share_id: str) -> HTMLResponse:
    """The read-only transcript page a share link opens.

    Server-rendered from the same store call the JSON endpoint uses, so the page needs no
    JavaScript and has nothing to fetch.
    """
    view = _store.shared_thread_view(share_id)

    if view is None:
        return HTMLResponse(content=render_not_found(), status_code=404, headers=SHARE_PAGE_HEADERS)

    return HTMLResponse(
        content=render_share_page(str(view.get("title") or ""), view.get("messages") or []),
        headers=SHARE_PAGE_HEADERS,
    )

@router.get("/chat/shared/{share_id}", response_model=SharedThreadResponse)
async def read_shared_thread(share_id: str) -> SharedThreadResponse:
    """Public and read-only: this is what the link is for.

    Deliberately unauthenticated, and deliberately unable to write. It returns the messages
    and nothing about the owner - no user id, no conversation id, no status.
    """
    view = _store.shared_thread_view(share_id)

    if view is None:
        raise HTTPException(status_code=404, detail="This share link does not exist")

    return SharedThreadResponse(
        title=str(view.get("title") or ""),
        messages=[
            SharedMessage(role=role, text=text)
            for role, text in (view.get("messages") or [])
        ],
    )
