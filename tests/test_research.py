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
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        HttpResearchProvider("file:///tmp/search")

    provider = HttpResearchProvider("https://example.com/search")
    with pytest.raises(ValueError, match="query"):
        asyncio.run(provider.search("   "))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(provider.search("indoone", limit=21))


def test_provider_sanitizes_result_urls(monkeypatch) -> None:
    class FakeResponse:
        content = json.dumps(
            {
                "results": [
                    {"title": "Valid", "url": "https://example.com", "snippet": "ok"},
                    {"title": "Bad scheme", "url": "javascript:alert(1)", "snippet": "skip"},
                    {"title": "Missing URL", "snippet": "skip"},
                ]
            }
        ).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return json.loads(self.content)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    results = asyncio.run(HttpResearchProvider("https://search.example/api").search("Indoone"))

    assert results == [ResearchResult("Valid", "https://example.com", "ok")]


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

def test_google_news_rss_provider_parses_sources(monkeypatch) -> None:
    rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>AI update</title>
        <link>https://example.com/a</link>
        <description><![CDATA[<p>Fresh <b>AI</b> evidence.</p>]]></description>
      </item>
      <item>
        <title>Second source</title>
        <link>https://example.org/b</link>
        <description>Another source.</description>
      </item>
    </channel></rss>"""

    class FakeResponse:
        content = rss
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def get(self, url, headers):
            assert "news.google.com/rss/search" in url
            assert "q=Indoone+AI" in url
            assert headers["Accept"].startswith("application/rss+xml")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    from app.ai.research import GoogleNewsRssResearchProvider
    results = asyncio.run(GoogleNewsRssResearchProvider().search("Indoone AI", limit=2))

    assert results[0] == ResearchResult("AI update", "https://example.com/a", "Fresh AI evidence.")
    assert results[1] == ResearchResult("Second source", "https://example.org/b", "Another source.")


def test_build_research_provider_uses_keyless_live_fallback(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_RESEARCH_URL", raising=False)
    from app.ai.research import GoogleNewsRssResearchProvider, build_research_provider

    assert isinstance(build_research_provider(), GoogleNewsRssResearchProvider)
