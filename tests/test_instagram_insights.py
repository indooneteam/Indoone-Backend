import pytest

from app.capabilities import instagram_insights


def test_validate_metrics_deduplicates_and_rejects_invalid() -> None:
    assert instagram_insights._validate_metrics(["reach", "reach", "followers_count"]) == ["reach", "followers_count"]
    with pytest.raises(ValueError):
        instagram_insights._validate_metrics(["reach-total"])
    with pytest.raises(ValueError):
        instagram_insights._validate_metrics([])


def test_validate_range_rejects_inverted_window() -> None:
    with pytest.raises(ValueError):
        instagram_insights._validate_range(200, 100)


@pytest.mark.anyio
async def test_account_insights_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_get_json(user_id: str, path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((user_id, params))
        assert path == "me/insights"
        return {"data": [{"name": "reach", "values": [{"value": 42}]}]}

    monkeypatch.setattr(instagram_insights, "_get_json_with_retry", fake_get_json)
    result = await instagram_insights.get_account_insights(
        "user-1",
        ["reach", "accounts_engaged", "reach"],
        "day",
        100,
        200,
    )

    assert calls == [
        (
            "user-1",
            {
                "metric": "reach,accounts_engaged",
                "period": "day",
                "since": 100,
                "until": 200,
            },
        )
    ]
    assert result["scope"] == "account"
    assert result["data"][0]["name"] == "reach"
    assert result["secrets_exposed"] is False


@pytest.mark.anyio
async def test_media_insights_reject_empty_media_id() -> None:
    with pytest.raises(ValueError, match="media_id is required"):
        await instagram_insights.get_media_insights("user-1", "", ["reach"])


@pytest.mark.anyio
async def test_media_insights_returns_provider_data(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_json(user_id: str, path: str, params: dict[str, object]) -> dict[str, object]:
        assert user_id == "user-1"
        assert path == "media-1/insights"
        assert params == {"metric": "reach,likes"}
        return {"data": [{"name": "likes", "values": [{"value": 7}]}]}

    monkeypatch.setattr(instagram_insights, "_get_json_with_retry", fake_get_json)
    result = await instagram_insights.get_media_insights("user-1", "media-1", ["reach", "likes"])

    assert result["scope"] == "media"
    assert result["media_id"] == "media-1"
    assert result["data"][0]["name"] == "likes"
    assert result["secrets_exposed"] is False
