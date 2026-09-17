from __future__ import annotations

import asyncio

import pytest

from app.capabilities import youtube_live


def test_broadcast_resource_validation_and_shape() -> None:
    payload = youtube_live.build_broadcast_resource(
        title="Launch Stream",
        scheduled_start_time="2026-10-30T19:00:00Z",
        privacy_status="unlisted",
        category_id="17",
        enable_dvr=True,
        latency_preference="low",
        self_declared_made_for_kids=False,
    )
    assert payload["snippet"] == {
        "title": "Launch Stream",
        "scheduledStartTime": "2026-10-30T19:00:00Z",
        "categoryId": "17",
    }
    assert payload["status"] == {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": False}
    assert payload["contentDetails"] == {"enableDvr": True, "latencyPreference": "low"}


def test_broadcast_resource_rejects_timezone_less_datetime() -> None:
    with pytest.raises(ValueError, match="must include a timezone"):
        youtube_live.build_broadcast_resource(
            title="Launch Stream",
            scheduled_start_time="2026-10-30T19:00:00",
        )


def test_stream_resource_shape() -> None:
    payload = youtube_live.build_stream_resource(
        title="Primary Stream",
        ingestion_type="rtmp",
        resolution="1080p",
        frame_rate="60fps",
        is_reusable=True,
    )
    assert payload["snippet"] == {"title": "Primary Stream"}
    assert payload["cdn"] == {
        "ingestionType": "rtmp",
        "resolution": "1080p",
        "frameRate": "60fps",
    }
    assert payload["contentDetails"] == {"isReusable": True}


def test_live_list_filters_are_validated_before_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scope(_: str) -> None:
        return None

    async def fake_request(*args, **kwargs):
        raise AssertionError("request must not be called for invalid filters")

    monkeypatch.setattr(youtube_live, "_ensure_live_scope", lambda _: None)
    monkeypatch.setattr(youtube_live, "_request", fake_request)
    with pytest.raises(ValueError, match="cannot be used together"):
        asyncio.run(
            youtube_live.list_broadcasts(
                "user-1",
                broadcast_id="abc",
                broadcast_status="upcoming",
            )
        )


def test_write_operations_require_explicit_approval() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(
            youtube_live.create_broadcast(
                "user-1",
                title="No Approval",
                scheduled_start_time="2026-10-30T19:00:00Z",
                approved=False,
            )
        )
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(
            youtube_live.transition_broadcast(
                "user-1",
                "broadcast-1",
                "live",
                approved=False,
            )
        )
    with pytest.raises(PermissionError, match="explicit approval"):
        asyncio.run(
            youtube_live.delete_live_chat_message(
                "user-1",
                "message-1",
                approved=False,
            )
        )


def test_stream_updates_are_metadata_only() -> None:
    with pytest.raises(ValueError, match="at least one of title or description"):
        asyncio.run(
            youtube_live.update_stream(
                "user-1",
                "stream-1",
                approved=True,
            )
        )
