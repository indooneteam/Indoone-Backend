from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

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


def test_main_registers_research_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            return []

    monkeypatch.setattr(api, "build_research_provider", lambda: EmptyProvider())

    from app.main import app

    client = TestClient(app)
    research_response = client.post("/api/research", json={"query": "latest AI news"})
    deep_response = client.post(
        "/api/deep-research",
        json={"query": "latest AI news", "queries": 2},
    )

    assert research_response.status_code == 200
    assert deep_response.status_code == 200
    assert research_response.json()["results"] == []
    assert deep_response.json()["sources"] == []
