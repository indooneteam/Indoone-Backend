from __future__ import annotations

import base64

import pytest

from app.capabilities.youtube_advanced import (
    parse_iso_duration,
    validate_short_eligibility,
)


def test_parse_iso_duration() -> None:
    assert parse_iso_duration("PT1M30S") == 90
    assert parse_iso_duration("PT3M") == 180
    assert parse_iso_duration("PT1H2M3S") == 3723


def test_parse_iso_duration_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        parse_iso_duration("90")


def test_short_eligibility_accepts_square_and_vertical_under_three_minutes() -> None:
    square = validate_short_eligibility(180, 1080, 1080)
    vertical = validate_short_eligibility(179.5, 1080, 1920)
    assert square["eligible"] is True
    assert vertical["eligible"] is True


def test_short_eligibility_rejects_landscape_and_over_limit() -> None:
    landscape = validate_short_eligibility(60, 1920, 1080)
    long_video = validate_short_eligibility(180.1, 1080, 1920)
    assert landscape["eligible"] is False
    assert long_video["eligible"] is False


def test_short_eligibility_reports_missing_dimensions() -> None:
    result = validate_short_eligibility(30, None, None)
    assert result["eligible"] is False
    assert "dimensions" in result["reason"]


@pytest.mark.asyncio
async def test_advanced_video_update_requires_explicit_approval() -> None:
    from app.capabilities.youtube_advanced import update_video_advanced

    with pytest.raises(PermissionError, match="explicit approval"):
        await update_video_advanced("user-1", "video-1", title="New title")


@pytest.mark.asyncio
async def test_caption_insert_requires_explicit_approval() -> None:
    from app.capabilities.youtube_advanced import insert_caption

    encoded = base64.b64encode(b"WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n").decode()
    with pytest.raises(PermissionError, match="explicit approval"):
        await insert_caption("user-1", "video-1", "en", "English", encoded)


@pytest.mark.asyncio
async def test_caption_delete_requires_explicit_approval() -> None:
    from app.capabilities.youtube_advanced import delete_caption

    with pytest.raises(PermissionError, match="explicit approval"):
        await delete_caption("user-1", "caption-1")


@pytest.mark.asyncio
async def test_caption_update_requires_content_or_draft() -> None:
    from app.capabilities.youtube_advanced import update_caption

    with pytest.raises(ValueError, match="content_base64 or is_draft"):
        await update_caption("user-1", "caption-1", approved=True)
