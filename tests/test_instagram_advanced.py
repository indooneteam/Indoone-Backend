import asyncio

import pytest

from app.capabilities import instagram_advanced


def test_validate_id_rejects_invalid_values() -> None:
    assert instagram_advanced._validate_id("media-1", "media_id") == "media-1"
    with pytest.raises(ValueError):
        instagram_advanced._validate_id("bad id", "media_id")
    with pytest.raises(ValueError):
        instagram_advanced._validate_id("", "creation_id")


def test_list_reels_filters_reel_media(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("u", "GET", "me/media")
        assert kwargs["params"] == {
            "fields": instagram_advanced._MEDIA_FIELDS,
            "limit": 10,
            "after": "next",
        }
        return {
            "data": [
                {"id": "r1", "media_product_type": "REELS", "media_type": "VIDEO"},
                {"id": "p1", "media_product_type": "FEED", "media_type": "IMAGE"},
            ],
            "paging": {"next": "x"},
        }

    monkeypatch.setattr(instagram_advanced, "_request_json", fake_request_json)
    result = asyncio.run(instagram_advanced.list_reels("u", 10, "next"))

    assert [item["id"] for item in result["data"]] == ["r1"]
    assert result["media_product_type"] == "REELS"
    assert result["secrets_exposed"] is False


def test_list_stories_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        assert (user_id, method, path) == ("u", "GET", "me/stories")
        assert kwargs["params"] == {
            "fields": instagram_advanced._STORY_FIELDS,
            "limit": 5,
        }
        return {"data": [{"id": "s1"}], "paging": None}

    monkeypatch.setattr(instagram_advanced, "_request_json", fake_request_json)
    result = asyncio.run(instagram_advanced.list_stories("u", 5))

    assert result["data"][0]["id"] == "s1"
    assert result["media_product_type"] == "STORIES"


def test_media_details_and_container_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request_json(user_id: str, method: str, path: str, **kwargs):
        return {"id": path, "status": "FINISHED"}

    monkeypatch.setattr(instagram_advanced, "_request_json", fake_request_json)
    media = asyncio.run(instagram_advanced.get_media_details("u", "m1"))
    container = asyncio.run(instagram_advanced.get_container_status("u", "c1"))

    assert media["media_id"] == "m1"
    assert container["creation_id"] == "c1"
    assert container["container"]["status"] == "FINISHED"
