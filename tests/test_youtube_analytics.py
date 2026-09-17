from __future__ import annotations

import pytest

import app.capabilities.youtube_analytics as analytics


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return self.response


def test_report_parameters_validate_dates_and_ids() -> None:
    params = analytics._build_report_params(
        start_date="2026-01-01",
        end_date="2026-01-31",
        metrics="views,likes",
        dimensions="day",
        sort="day",
        max_results=20,
    )
    assert params["ids"] == "channel==MINE"
    assert params["startDate"] == "2026-01-01"
    assert params["endDate"] == "2026-01-31"
    assert params["metrics"] == "views,likes"
    assert params["dimensions"] == "day"
    with pytest.raises(ValueError, match="on or after"):
        analytics._build_report_params(
            start_date="2026-02-01",
            end_date="2026-01-31",
            metrics="views",
        )
    with pytest.raises(ValueError, match="channel==MINE"):
        analytics._build_report_params(
            start_date="2026-01-01",
            end_date="2026-01-31",
            metrics="views",
            ids="channel==other",
        )


def test_scope_validation_requires_both_read_scopes() -> None:
    with pytest.raises(PermissionError, match="youtube.readonly"):
        analytics._validate_granted_scopes(analytics._ANALYTICS_READ_SCOPE)
    assert analytics._validate_granted_scopes(
        f"{analytics._YOUTUBE_READ_SCOPE} {analytics._ANALYTICS_READ_SCOPE}"
    )


@pytest.mark.asyncio
async def test_query_report_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_access_token(user_id: str) -> str:
        return "token"

    fake_client = FakeClient(
        FakeResponse(
            {
                "columnHeaders": [
                    {"name": "day", "columnType": "DIMENSION"},
                    {"name": "views", "columnType": "METRIC"},
                ],
                "rows": [["2026-01-01", 12]],
            }
        )
    )
    monkeypatch.setattr(analytics, "_access_token", fake_access_token)
    monkeypatch.setattr(analytics.httpx, "AsyncClient", lambda **kwargs: fake_client)

    result = await analytics.query_report(
        "user-1",
        start_date="2026-01-01",
        end_date="2026-01-31",
        metrics="views",
        dimensions="day",
        sort="day",
        max_results=10,
    )

    assert result["row_count"] == 1
    assert result["rows_as_objects"] == [{"day": "2026-01-01", "views": 12}]
    assert fake_client.calls[0][2]["params"] == {
        "ids": "channel==MINE",
        "startDate": "2026-01-01",
        "endDate": "2026-01-31",
        "metrics": "views",
        "maxResults": 10,
        "startIndex": 1,
        "dimensions": "day",
        "sort": "day",
    }


@pytest.mark.asyncio
async def test_daily_trend_uses_day_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    async def fake_query_report(user_id: str, **kwargs: object) -> dict[str, object]:
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(analytics, "query_report", fake_query_report)
    result = await analytics.daily_channel_trend("user-1", "2026-01-01", "2026-01-31", max_results=31)

    assert result == {"ok": True}
    assert called["dimensions"] == "day"
    assert called["sort"] == "day"
    assert called["max_results"] == 31
