from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.ai.memory_policy import MemoryPolicy
from app.ai.memory_service import MemoryService
from app.api.dependencies import current_user_id

router = APIRouter(tags=["memory"])
_memory_service = MemoryService(policy=MemoryPolicy(allow_automatic_write=False))


class MemoryWriteRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="user", min_length=1, max_length=128)


class MemoryDeleteRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=128)


@router.get("/memory")
async def get_memories(
    request: Request,
    key_prefix: str = Query(default="", max_length=128),
    source: str = Query(default="", max_length=128),
    limit: int = Query(default=100, ge=1, le=100),
    q: str = Query(default="", max_length=500),
) -> dict[str, object]:
    user_id = current_user_id(request)
    if q.strip():
        return {"memories": _memory_service.search(user_id, q, limit=limit)}

    from app.capabilities.store import list_memories

    return {
        "memories": list_memories(
            user_id,
            key_prefix=key_prefix.strip(),
            source=source.strip(),
            limit=limit,
        )
    }


@router.post("/memory")
async def write_memory(request: Request, body: MemoryWriteRequest) -> dict[str, object]:
    user_id = current_user_id(request)
    try:
        memory = _memory_service.write_candidate(
            user_id,
            key=body.key,
            value=body.value,
            confidence=body.confidence,
            source=body.source,
            explicit_user_request=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(status_code=400, detail="memory write rejected by policy")
    return {"memory": memory}


@router.delete("/memory")
async def remove_memory(request: Request, body: MemoryDeleteRequest) -> dict[str, bool]:
    user_id = current_user_id(request)
    if not _memory_service.delete(user_id, body.memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": True}
