from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from secrets import token_urlsafe

from app.capabilities.canva import (
    build_canva_authorization,
    create_export_job,
    exchange_canva_code,
    get_design,
    get_export_formats,
    get_export_job,
    list_designs,
)
from app.capabilities.store import create_oauth_state

router = APIRouter(prefix="/integrations/canva", tags=["canva"])


class ConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)
    include_write: bool = True


class CallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)
    code_verifier: str = Field(min_length=43, max_length=256)


class UserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class DesignsRequest(UserRequest):
    query: str = Field(default="", max_length=500)
    continuation: str = Field(default="", max_length=2048)


class DesignRequest(UserRequest):
    design_id: str = Field(min_length=1, max_length=256)


class ExportCreateRequest(UserRequest):
    design_id: str = Field(min_length=1, max_length=256)
    format: dict[str, object]
    approved: bool = False


class ExportJobRequest(UserRequest):
    export_id: str = Field(min_length=1, max_length=256)


@router.post("/connect")
async def connect(request: ConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "canva", request.redirect_uri.strip())
        return {**build_canva_authorization(state, request.redirect_uri, request.include_write), "state": state}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/callback")
async def callback(request: CallbackRequest) -> dict[str, object]:
    try:
        return await exchange_canva_code(request.state, request.code, request.code_verifier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"canva oauth failed: {exc}") from exc


@router.post("/designs")
async def designs(request: DesignsRequest) -> dict[str, object]:
    try:
        return await list_designs(request.user_id, request.query, request.continuation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"canva provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/design")
async def design(request: DesignRequest) -> dict[str, object]:
    try:
        return await get_design(request.user_id, request.design_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"canva provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/export-formats")
async def export_formats(request: DesignRequest) -> dict[str, object]:
    try:
        return await get_export_formats(request.user_id, request.design_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"canva provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/export")
async def export_design(request: ExportCreateRequest) -> dict[str, object]:
    try:
        return await create_export_job(request.user_id, request.design_id, request.format, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"canva provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/export-job")
async def export_job(request: ExportJobRequest) -> dict[str, object]:
    try:
        return await get_export_job(request.user_id, request.export_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"canva provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
