import pytest

from app.capabilities.google_calendar import create_event, delete_event, list_calendars, list_events, update_event


@pytest.mark.asyncio
async def test_calendar_create_requires_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        await create_event("user-1", "primary", {"start": {"dateTime": "2026-09-13T10:00:00+05:30"}, "end": {"dateTime": "2026-09-13T11:00:00+05:30"}}, False)


@pytest.mark.asyncio
async def test_calendar_update_requires_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        await update_event("user-1", "primary", "event-1", {"summary": "Changed"}, False)


@pytest.mark.asyncio
async def test_calendar_delete_requires_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        await delete_event("user-1", "primary", "event-1", False)


@pytest.mark.asyncio
async def test_calendar_list_calendars_validates_page_size_before_provider_access() -> None:
    with pytest.raises(ValueError, match="max_results"):
        await list_calendars("user-1", max_results=0)


@pytest.mark.asyncio
async def test_calendar_list_events_validates_page_size_before_provider_access() -> None:
    with pytest.raises(ValueError, match="max_results"):
        await list_events("user-1", max_results=0)
