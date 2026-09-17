from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.phone import build_call_action

router = APIRouter(tags=["phone"])


class CallRequest(BaseModel):
    phone_number: str = Field(min_length=7, max_length=32)
    contact_name: str = Field(default="", max_length=200)


@router.post("/phone/call")
async def request_call(request: CallRequest) -> dict[str, object]:
    try:
        action = build_call_action(request.phone_number, request.contact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"action": action.as_dict()}
