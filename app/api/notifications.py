from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.api.dependencies import current_user_id
from app.notifications.store import register_token


router = APIRouter(tags=["notifications"])


class RegisterNotificationTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


@router.post("/notifications/register")
async def register_notification_token(
    request: Request,
    body: RegisterNotificationTokenRequest,
) -> dict[str, object]:
    user_id = current_user_id(request)
    register_token(user_id=user_id, token=body.token)
    return {"registered": True}
