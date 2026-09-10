import asyncio
import json

import httpx
import pytest

from app.ai.research import HttpResearchProvider, ResearchResult, format_results


def test_format_results_preserves_provenance() -> None:
    formatted = format_results(
        [ResearchResult("Example", "https://example.com", "A useful snippet")]
    )
    assert "<research>" in formatted
    assert "title: Example" in formatted
    assert "url: https://example.com" in formatted
    assert "snippet: A useful snippet" in formatted


def test_provider_validates_inputs() -> None:
    provider = HttpResearchProvider("https://example.com/search")
    with pytest.raises(ValueError, match="query"):
        asyncio.run(provider.search("   "))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(provider.search("indoone", limit=21))


def test_provider_parses_json_results(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeResponse:
        content = json.dumps(
            {"results": [{"title": "Indoone", "url": "https://indoone.ai", "snippet": "local AI"}]}
        ).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return json.loads(self.content)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            captured["url"] = url
            captured["auth"] = headers.get("Authorization", "")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    provider = HttpResearchProvider(
        "https://search.example/api",
        bearer_token="secret",
    )
    results = asyncio.run(provider.search("Indoone AI"))

    assert results == [ResearchResult("Indoone", "https://indoone.ai", "local AI")]
    assert "q=Indoone+AI" in captured["url"]
    assert captured["auth"] == "Bearer secret"
