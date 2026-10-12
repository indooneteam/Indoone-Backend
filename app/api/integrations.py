from __future__ import annotations

import base64
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.gmail.routes import router as gmail_router
from app.plugins.github.routes import router as github_router
from app.capabilities.google_calendar import create_event as create_calendar_event, delete_event as delete_calendar_event, get_event as get_calendar_event, list_calendars, list_events, update_event as update_calendar_event
from app.capabilities.google_drive import (
    build_google_drive_authorization,
    exchange_google_drive_code,
    get_drive_file,
    list_drive_files,
    probe_google_drive,
    upload_drive_file,
)
from app.capabilities.integrations import (
    probe_integration,
)
from app.capabilities.store import create_oauth_state
from secrets import token_urlsafe

router = APIRouter(prefix="/integrations", tags=["integrations"])
router.include_router(gmail_router)
router.include_router(github_router)


class IntegrationProbeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


class GoogleCalendarListRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    page_token: str = Field(default="", max_length=2048)
    max_results: int = Field(default=100, ge=1, le=250)


class GoogleCalendarEventsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    calendar_id: str = Field(default="primary", min_length=1, max_length=512)
    time_min: str = Field(default="", max_length=100)
    time_max: str = Field(default="", max_length=100)
    query: str = Field(default="", max_length=500)
    page_token: str = Field(default="", max_length=2048)
    max_results: int = Field(default=100, ge=1, le=2500)


class GoogleCalendarEventRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    calendar_id: str = Field(default="primary", min_length=1, max_length=512)
    event_id: str = Field(default="", max_length=1024)


class GoogleCalendarWriteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    calendar_id: str = Field(default="primary", min_length=1, max_length=512)
    event_id: str = Field(default="", max_length=1024)
    event: dict[str, object]
    approved: bool = False


class GoogleCalendarDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    calendar_id: str = Field(default="primary", min_length=1, max_length=512)
    event_id: str = Field(min_length=1, max_length=1024)
    approved: bool = False


class GoogleDriveConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2000)


class GoogleDriveCallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8000)


class GoogleDriveFilesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    query: str = Field(default="", max_length=500)
    page_token: str = Field(default="", max_length=2048)
    page_size: int = Field(default=50, ge=1, le=1000)


class GoogleDriveFileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    file_id: str = Field(min_length=1, max_length=512)
    download: bool = False
    export_mime_type: str = Field(default="", max_length=256)


class GoogleDriveUploadRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    filename: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=1, max_length=256)
    content_base64: str = Field(min_length=1, max_length=20_000_000)
    parent_id: str = Field(default="", max_length=512)
    approved: bool = False


@router.post("/{integration_id}/probe")
async def integration_probe(integration_id: str, request: IntegrationProbeRequest) -> dict[str, object]:
    try:
        if integration_id.strip().lower() == "google_drive":
            return await probe_google_drive(request.user_id)
        return await probe_integration(request.user_id, integration_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"integration provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/calendars")
async def google_calendar_calendars(request: GoogleCalendarListRequest) -> dict[str, object]:
    try:
        return await list_calendars(request.user_id, request.page_token, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/events")
async def google_calendar_events(request: GoogleCalendarEventsRequest) -> dict[str, object]:
    try:
        return await list_events(request.user_id, request.calendar_id, request.time_min, request.time_max, request.query, request.page_token, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/event")
async def google_calendar_event(request: GoogleCalendarEventRequest) -> dict[str, object]:
    try:
        return await get_calendar_event(request.user_id, request.calendar_id, request.event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/create-event")
async def google_calendar_create_event(request: GoogleCalendarWriteRequest) -> dict[str, object]:
    try:
        return await create_calendar_event(request.user_id, request.calendar_id, request.event, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/update-event")
async def google_calendar_update_event(request: GoogleCalendarWriteRequest) -> dict[str, object]:
    try:
        return await update_calendar_event(request.user_id, request.calendar_id, request.event_id, request.event, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-calendar/delete-event")
async def google_calendar_delete_event(request: GoogleCalendarDeleteRequest) -> dict[str, object]:
    try:
        return await delete_calendar_event(request.user_id, request.calendar_id, request.event_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google calendar provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-drive/connect")
async def google_drive_connect(request: GoogleDriveConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, "google_drive", request.redirect_uri.strip())
        return {**build_google_drive_authorization(state, request.redirect_uri), "state": state}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-drive/callback")
async def google_drive_callback(request: GoogleDriveCallbackRequest) -> dict[str, object]:
    try:
        return await exchange_google_drive_code(request.state, request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=f"google drive oauth failed: {exc}") from exc


@router.post("/google-drive/files")
async def google_drive_files(request: GoogleDriveFilesRequest) -> dict[str, object]:
    try:
        return await list_drive_files(request.user_id, request.query, request.page_token, request.page_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google drive provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-drive/file")
async def google_drive_file(request: GoogleDriveFileRequest) -> dict[str, object]:
    try:
        return await get_drive_file(request.user_id, request.file_id, request.download, request.export_mime_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google drive provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/google-drive/upload")
async def google_drive_upload(request: GoogleDriveUploadRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        return await upload_drive_file(request.user_id, request.filename, request.mime_type, content, request.parent_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"google drive provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


