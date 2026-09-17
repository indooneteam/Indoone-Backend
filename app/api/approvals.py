from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.approval import issue_approval_token
from app.ai.tool_registry import get_tool_spec
from app.api.dependencies import current_user_id

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalRequest(BaseModel):
    tool: str = Field(min_length=1, max_length=128)
    ttl_seconds: int = Field(default=300, ge=1, le=3600)


class ApprovalResponse(BaseModel):
    tool: str
    expires_in_seconds: int
    approval_token: str


@router.post("/issue", response_model=ApprovalResponse)
async def issue_approval(request: Request, body: ApprovalRequest) -> ApprovalResponse:
    user_id = current_user_id(request)
    tool = body.tool.strip().lower()
    spec = get_tool_spec(tool)
    if spec is None:
        raise HTTPException(status_code=404, detail="tool is not registered")
    if not spec.requires_approval:
        raise HTTPException(status_code=400, detail="tool does not require approval")
    try:
        token = issue_approval_token(user_id, tool, ttl_seconds=body.ttl_seconds)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApprovalResponse(tool=tool, expires_in_seconds=body.ttl_seconds, approval_token=token)
