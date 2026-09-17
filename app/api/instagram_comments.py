from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_comments import (
    create_media_comment,
    delete_comment,
    list_media_comments,
    reply_to_comment,
    set_comment_hidden,
)

router = APIRouter(prefix="/integrations/instagram/comments", tags=["instagram-comments"])


class MediaCommentsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    media_id: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=25, ge=1, le=100)
    after: str = Field(default="", max_length=2048)


class CommentRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    comment_id: str = Field(min_length=1, max_length=256)
    message: str = Field(default="", max_length=2200)
    approved: bool = False


class CreateCommentRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    media_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=2200)
    approved: bool = False


class ModerationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    comment_id: str = Field(min_length=1, max_length=256)
    hidden: bool
    approved: bool = False


@router.post("/media")
async def media_comments(request: MediaCommentsRequest) -> dict[str, object]:
    try:
        return await list_media_comments(request.user_id, request.media_id, request.limit, request.after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram comments provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/create")
async def create_comment(request: CreateCommentRequest) -> dict[str, object]:
    try:
        return await create_media_comment(request.user_id, request.media_id, request.message, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram comments provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/reply")
async def reply(request: CommentRequest) -> dict[str, object]:
    try:
        return await reply_to_comment(request.user_id, request.comment_id, request.message, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram comments provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/delete")
async def delete(request: CommentRequest) -> dict[str, object]:
    try:
        return await delete_comment(request.user_id, request.comment_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram comments provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/moderate")
async def moderate(request: ModerationRequest) -> dict[str, object]:
    try:
        return await set_comment_hidden(request.user_id, request.comment_id, request.hidden, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram comments provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
