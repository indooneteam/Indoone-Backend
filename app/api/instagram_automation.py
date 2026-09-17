from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_automation import plan_automation
from app.capabilities.instagram_comments import reply_to_comment
from app.capabilities.instagram_messaging import send_text_message

router = APIRouter(prefix="/integrations/instagram/automation", tags=["instagram-automation"])


class Rule(BaseModel):
    trigger: str = Field(min_length=1, max_length=32)
    action: str = Field(min_length=1, max_length=32)
    keyword: str = Field(default="", max_length=2200)
    response: str = Field(min_length=1, max_length=2200)
    enabled: bool = True


class PlanRequest(BaseModel):
    payload: dict[str, object]
    rules: list[Rule] = Field(min_length=1, max_length=50)


class Action(BaseModel):
    action: str
    comment_id: str = ""
    recipient_id: str = ""
    response: str = Field(min_length=1, max_length=2200)


class ExecuteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    actions: list[Action] = Field(min_length=1, max_length=50)
    approved: bool = False


@router.post("/plan")
async def plan(request: PlanRequest) -> dict[str, object]:
    try:
        return plan_automation(request.payload, [rule.model_dump() for rule in request.rules])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/execute")
async def execute(request: ExecuteRequest) -> dict[str, object]:
    if not request.approved:
        raise HTTPException(status_code=403, detail="instagram automation execution requires explicit approval")

    results: list[dict[str, object]] = []
    for action in request.actions:
        try:
            normalized_action = action.action.strip().lower()
            if normalized_action == "reply_comment":
                result = await reply_to_comment(
                    request.user_id,
                    action.comment_id,
                    action.response,
                    approved=True,
                )
            elif normalized_action == "send_message":
                result = await send_text_message(
                    request.user_id,
                    action.recipient_id,
                    action.response,
                    approved=True,
                )
            else:
                raise ValueError("unsupported instagram automation action")
            results.append({"action": normalized_action, "success": True, "result": result})
        except (ValueError, PermissionError) as exc:
            results.append({"action": action.action, "success": False, "error": str(exc)})
        except httpx.HTTPError as exc:
            results.append({"action": action.action, "success": False, "error": f"provider failed: {exc}"})
        except RuntimeError as exc:
            results.append({"action": action.action, "success": False, "error": str(exc)})

    return {
        "integration": "instagram",
        "executed": True,
        "results": results,
        "secrets_exposed": False,
    }
