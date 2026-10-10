from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.web_research import api
from app.web_research.research import ResearchResult


def test_research_endpoint_returns_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest AI news"
            assert limit == 2
            return [ResearchResult("Official update", "https://example.com/update", "Fresh evidence")]

    monkeypatch.setattr(api, "build_research_provider", lambda: FakeProvider())

    response = asyncio.run(api.research(api.ResearchRequest(query="latest AI news", limit=2)))

    assert response == {
        "query": "latest AI news",
        "results": [
            {
                "title": "Official update",
                "url": "https://example.com/update",
                "snippet": "Fresh evidence",
            }
        ],
    }


def test_deep_research_endpoint_deduplicates_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            if query == "Indoone":
                return [ResearchResult("Source A", "https://example.com/a", "One")]
            return [
                ResearchResult("Source A revised", "https://example.com/a", "Longer"),
                ResearchResult("Source B", "https://example.com/b", "Two"),
            ]

    monkeypatch.setattr(api, "build_research_provider", lambda: FakeProvider())

    response = asyncio.run(
        api.deep_research(api.DeepResearchRequest(query="Indoone", queries=2, per_query_limit=2))
    )

    assert response["queries"] == ["Indoone", "Indoone official sources"]
    assert [item["url"] for item in response["sources"]] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_research_endpoint_returns_503_when_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "build_research_provider", lambda: None)

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.research(api.ResearchRequest(query="latest AI news")))

    assert error.value.status_code == 503


def test_main_registers_research_routes() -> None:
    from app.main import app

    paths = set(app.openapi()["paths"])
    assert "/api/research" in paths
    assert "/api/deep-research" in paths
