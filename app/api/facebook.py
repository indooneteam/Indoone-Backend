from __future__ import annotations

from secrets import token_urlsafe

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.facebook import (
    build_facebook_authorization,
    create_page_post,
    exchange_facebook_code,
    list_page_posts,
    list_pages,
)
from app.capabilities.store import create_oauth_state

router = APIRouter(prefix="/integrations/facebook", tags=["facebook"])


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)


class CallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class UserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class PagePostsRequest(UserRequest):
    page_id: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=25, ge=1, le=100)
    after: str = Field(default="", max_length=4096)


class CreatePostRequest(UserRequest):
    page_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=63206)
    approved: bool = False


@router.post("/connect")
async def connect(request: ConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "facebook", request.redirect_uri.strip())
        return {**build_facebook_authorization(state, request.redirect_uri), "state": state}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: CallbackRequest) -> dict[str, object]:
    try:
        return await exchange_facebook_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"facebook oauth failed: {exc}") from exc


@router.post("/pages")
async def pages(request: UserRequest) -> dict[str, object]:
    try:
        return await list_pages(request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"facebook provider failed: {exc}") from exc


@router.post("/posts")
async def posts(request: PagePostsRequest) -> dict[str, object]:
    try:
        return await list_page_posts(request.user_id, request.page_id, request.limit, request.after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"facebook provider failed: {exc}") from exc


@router.post("/publish")
async def publish(request: CreatePostRequest) -> dict[str, object]:
    try:
        return await create_page_post(request.user_id, request.page_id, request.message, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"facebook provider failed: {exc}") from exc
