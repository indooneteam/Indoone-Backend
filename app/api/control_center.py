from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from app.capabilities.control_center import (
    get_control_activity,
    get_control_metrics,
    get_control_state,
    is_control_center_admin_authorization,
    update_control_settings,
)

router = APIRouter(prefix="/control-center", tags=["control-center"])


def _require_admin(request: Request) -> None:
    if not is_control_center_admin_authorization(request.headers.get("authorization", "")):
        raise HTTPException(status_code=401, detail="Control Center admin authorization required")


@router.get("/status")
async def control_center_status(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return get_control_state()


@router.patch("/settings")
async def control_center_update_settings(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    _require_admin(request)
    try:
        return update_control_settings(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/metrics")
async def control_center_metrics(
    request: Request,
    window_hours: int = Query(default=24, ge=1, le=720),
) -> dict[str, Any]:
    _require_admin(request)
    return get_control_metrics(window_hours=window_hours)


@router.get("/activity")
async def control_center_activity(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    _require_admin(request)
    return get_control_activity(limit=limit)
