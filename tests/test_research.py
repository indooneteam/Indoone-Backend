import asyncio
import json

import httpx
import pytest

from app.ai.research import (
    CrossrefResearchProvider,
    GoogleNewsRssResearchProvider,
    HttpResearchProvider,
    MultiSourceResearchProvider,
    OpenAlexResearchProvider,
    ResearchResult,
    WikidataResearchProvider,
    WikipediaResearchProvider,
    build_research_query_variants,
    format_results,
)


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


def test_build_research_provider_uses_multi_source_defaults(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_RESEARCH_URL", raising=False)
    from app.ai.research import (
        CrossrefResearchProvider,
        GoogleNewsRssResearchProvider,
        MultiSourceResearchProvider,
        OpenAlexResearchProvider,
        WikidataResearchProvider,
        WikipediaResearchProvider,
        build_research_provider,
    )

    provider = build_research_provider()
    assert isinstance(provider, MultiSourceResearchProvider)
    assert {type(item) for item in provider.providers} == {
        WikipediaResearchProvider,
        WikidataResearchProvider,
        GoogleNewsRssResearchProvider,
        OpenAlexResearchProvider,
        CrossrefResearchProvider,
    }


def test_build_research_query_variants_extracts_latin_terms() -> None:
    variants = build_research_query_variants(
        "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ simple ಆಗಿ ಹೇಳು ಮತ್ತು sources ಕೊಡು."
    )

    assert variants[0].startswith("ಈಗಿನ AI technology")
    assert variants[1] == "AI technology research simple sources"


def test_wikipedia_research_provider_adapts_knowledge_answer(monkeypatch) -> None:
    from app.ai.general_knowledge import WikipediaAnswer

    class FakeWikipedia:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def answer(self, query):
            assert query == "Artificial intelligence"
            return WikipediaAnswer(
                "Artificial intelligence",
                "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "AI is machine intelligence.",
            )

    import app.ai.general_knowledge as general_knowledge
    monkeypatch.setattr(general_knowledge, "WikipediaKnowledgeProvider", FakeWikipedia)

    results = asyncio.run(
        WikipediaResearchProvider().search("Artificial intelligence", limit=2)
    )

    assert results == [
        ResearchResult(
            "Artificial intelligence",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "AI is machine intelligence.",
        )
    ]


def test_wikidata_provider_parses_entities(monkeypatch) -> None:
    payload = {
        "search": [
            {"id": "Q11660", "label": "Artificial intelligence", "description": "field of study"},
            {"id": "Q999", "label": "Other", "description": "another result"},
        ]
    }

    class FakeResponse:
        content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params, headers):
            assert "wikidata.org/w/api.php" in url
            assert params["action"] == "wbsearchentities"
            assert params["language"] == "en"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    results = asyncio.run(WikidataResearchProvider().search("Artificial intelligence", limit=2))

    assert results[0] == ResearchResult(
        "Artificial intelligence",
        "https://www.wikidata.org/wiki/Q11660",
        "field of study",
    )


def test_openalex_provider_parses_works(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "A research paper",
                "doi": "https://doi.org/10.1234/example",
                "publication_year": 2026,
                "abstract_inverted_index": {
                    "Research": [1],
                    "AI": [0],
                    "works": [2],
                },
            }
        ]
    }

    class FakeResponse:
        content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params, headers):
            assert "api.openalex.org/works" in url
            assert params["search"] == "AI"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    results = asyncio.run(OpenAlexResearchProvider().search("AI", limit=1))

    assert results[0] == ResearchResult(
        "A research paper",
        "https://doi.org/10.1234/example",
        "AI Research works",
    )


def test_crossref_provider_parses_works(monkeypatch) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "title": ["Published AI work"],
                    "URL": "https://doi.org/10.5555/example",
                    "abstract": "<jats:p>Fresh AI evidence.</jats:p>",
                }
            ]
        }
    }

    class FakeResponse:
        content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params, headers):
            assert "api.crossref.org/works" in url
            assert params["query"] == "AI"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    results = asyncio.run(CrossrefResearchProvider().search("AI", limit=1))

    assert results[0] == ResearchResult(
        "Published AI work",
        "https://doi.org/10.5555/example",
        "Fresh AI evidence.",
    )


def test_multi_source_provider_merges_sources_and_queries() -> None:
    class FakeProvider:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            self.queries: list[str] = []

        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            self.queries.append(query)
            return [
                ResearchResult(
                    f"{self.domain} result",
                    f"https://{self.domain}/result",
                    query,
                )
            ]

    wikipedia = FakeProvider("wikipedia.example")
    news = FakeProvider("news.example")
    provider = MultiSourceResearchProvider(
        [wikipedia, news],
        max_query_variants=2,
    )

    results = asyncio.run(
        provider.search(
            "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ",
            limit=3,
        )
    )

    assert len(results) == 2
    assert wikipedia.queries == [
        "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ",
        "AI technology research",
    ]
    assert news.queries == wikipedia.queries
    assert {item.url for item in results} == {
        "https://wikipedia.example/result",
        "https://news.example/result",
    }
