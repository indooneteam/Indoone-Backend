from __future__ import annotations

import os
from typing import Any

import httpx
from cryptography.fernet import Fernet

from app.capabilities.store import get_integration_token

_BASE = "https://www.googleapis.com/calendar/v3"


def _token(user_id: str) -> str:
    row = get_integration_token(user_id.strip(), "google_calendar")
    if row is None:
        raise ValueError("integration is not connected for user")
    key = os.getenv("INDOONE_OAUTH_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("oauth encryption key is not configured")
    try:
        return Fernet(key.encode("ascii")).decrypt(bytes(row["access_token"])).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored oauth token cannot be decrypted") from exc


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def list_calendars(user_id: str, page_token: str = "", max_results: int = 100) -> dict[str, object]:
    if not 1 <= max_results <= 250:
        raise ValueError("max_results must be between 1 and 250")
    params: dict[str, object] = {"maxResults": max_results}
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_BASE}/users/me/calendarList", headers=_headers(_token(user_id)), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google calendar returned invalid calendar list")
    return {"integration": "google_calendar", "user_id": user_id.strip(), "calendars": body.get("items", []), "next_page_token": body.get("nextPageToken"), "secrets_exposed": False}


async def list_events(user_id: str, calendar_id: str = "primary", time_min: str = "", time_max: str = "", query: str = "", page_token: str = "", max_results: int = 100) -> dict[str, object]:
    if not calendar_id.strip() or len(calendar_id) > 512:
        raise ValueError("calendar_id is required")
    if not 1 <= max_results <= 2500:
        raise ValueError("max_results must be between 1 and 2500")
    params: dict[str, object] = {"maxResults": max_results, "singleEvents": True, "orderBy": "startTime"}
    if time_min.strip():
        params["timeMin"] = time_min.strip()
    if time_max.strip():
        params["timeMax"] = time_max.strip()
    if query.strip():
        params["q"] = query.strip()
    if page_token.strip():
        params["pageToken"] = page_token.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_BASE}/calendars/{calendar_id.strip()}/events", headers=_headers(_token(user_id)), params=params)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("google calendar returned invalid event list")
    return {"integration": "google_calendar", "user_id": user_id.strip(), "calendar_id": calendar_id.strip(), "events": body.get("items", []), "next_page_token": body.get("nextPageToken"), "secrets_exposed": False}


async def get_event(user_id: str, calendar_id: str, event_id: str) -> dict[str, object]:
    calendar_id, event_id = calendar_id.strip(), event_id.strip()
    if not calendar_id or not event_id:
        raise ValueError("calendar_id and event_id are required")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{_BASE}/calendars/{calendar_id}/events/{event_id}", headers=_headers(_token(user_id)))
        response.raise_for_status()
        body = response.json()
    return {"integration": "google_calendar", "user_id": user_id.strip(), "calendar_id": calendar_id, "event": body, "secrets_exposed": False}


async def create_event(user_id: str, calendar_id: str, event: dict[str, Any], approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for calendar create operations")
    if not isinstance(event, dict) or not isinstance(event.get("start"), dict) or not isinstance(event.get("end"), dict):
        raise ValueError("event must include start and end objects")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{_BASE}/calendars/{calendar_id.strip()}/events", headers={**_headers(_token(user_id)), "Content-Type": "application/json"}, json=event)
        response.raise_for_status()
        body = response.json()
    return {"integration": "google_calendar", "user_id": user_id.strip(), "operation": "create", "event": body, "secrets_exposed": False}


async def update_event(user_id: str, calendar_id: str, event_id: str, event: dict[str, Any], approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for calendar update operations")
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.patch(f"{_BASE}/calendars/{calendar_id.strip()}/events/{event_id.strip()}", headers={**_headers(_token(user_id)), "Content-Type": "application/json"}, json=event)
        response.raise_for_status()
        body = response.json()
    return {"integration": "google_calendar", "user_id": user_id.strip(), "operation": "update", "event": body, "secrets_exposed": False}


async def delete_event(user_id: str, calendar_id: str, event_id: str, approved: bool = False) -> dict[str, object]:
    if not approved:
        raise PermissionError("explicit approval is required for calendar delete operations")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.delete(f"{_BASE}/calendars/{calendar_id.strip()}/events/{event_id.strip()}", headers=_headers(_token(user_id)))
        response.raise_for_status()
    return {"integration": "google_calendar", "user_id": user_id.strip(), "operation": "delete", "calendar_id": calendar_id.strip(), "event_id": event_id.strip(), "deleted": True, "secrets_exposed": False}
