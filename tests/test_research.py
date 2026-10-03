import asyncio
import json

import httpx
import pytest

from app.ai.research import (
    CrossrefResearchProvider,
    TavilyResearchProvider,
    GoogleNewsRssResearchProvider,
    HttpResearchProvider,
    MultiSourceResearchProvider,
    OpenAlexResearchProvider,
    ResearchResult,
    SearXNGResearchProvider,
    _is_relevant_research_result,
    WikidataResearchProvider,
    WikipediaResearchProvider,
    build_research_query_variants,
    format_research_context,
    format_results,
    score_research_result,
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

def test_tavily_provider_parses_ranked_results(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "title": "What is gravity?",
                "url": "https://example.com/gravity",
                "content": "Gravity is the force that attracts masses.",
                "score": 0.97,
            },
            {
                "title": "Gravity overview",
                "url": "https://example.org/gravity",
                "content": "Objects with mass attract one another.",
            },
        ]
    }

    class FakeResponse:
        content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            assert kwargs["follow_redirects"] is False
            assert kwargs["headers"]["Accept"] == "application/json"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            assert url == "https://api.tavily.com/search"
            assert json["api_key"] == "test-key"
            assert json["query"] == "What is gravity?"
            assert json["search_depth"] == "basic"
            assert json["topic"] == "general"
            assert json["max_results"] == 2
            assert json["include_answer"] is False
            assert json["include_raw_content"] is False
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    results = asyncio.run(
        TavilyResearchProvider("test-key").search("What is gravity?", limit=2)
    )

    assert results == [
        ResearchResult(
            "What is gravity?",
            "https://example.com/gravity",
            "Gravity is the force that attracts masses.",
        ),
        ResearchResult(
            "Gravity overview",
            "https://example.org/gravity",
            "Objects with mass attract one another.",
        ),
    ]


def test_tavily_provider_validates_inputs() -> None:
    with pytest.raises(ValueError, match="api_key"):
        TavilyResearchProvider("   ")

    provider = TavilyResearchProvider("test-key")
    with pytest.raises(ValueError, match="query"):
        asyncio.run(provider.search("   "))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(provider.search("gravity", limit=21))


def test_build_research_provider_uses_tavily_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("INDOONE_RESEARCH_URL", raising=False)

    provider = __import__("app.ai.research", fromlist=["build_research_provider"]).build_research_provider()

    assert isinstance(provider, MultiSourceResearchProvider)
    assert [type(item) for item in provider.providers] == [TavilyResearchProvider]


def test_searxng_provider_parses_json_results(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "title": "Fresh AI result",
                "url": "https://example.com/ai",
                "content": "Current AI evidence.",
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
        def __init__(self, *args, **kwargs) -> None:
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params):
            assert url == "http://127.0.0.1:8888/search"
            assert params["q"] == "latest AI news"
            assert params["format"] == "json"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    provider = SearXNGResearchProvider("http://127.0.0.1:8888")
    results = asyncio.run(provider.search("latest AI news", limit=1))

    assert results == [
        ResearchResult(
            "Fresh AI result",
            "https://example.com/ai",
            "Current AI evidence.",
        )
    ]


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


def test_build_research_provider_uses_multi_search(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_SEARCH_PROVIDER", "multi")
    monkeypatch.setenv("INDOONE_SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.delenv("INDOONE_RESEARCH_URL", raising=False)
    monkeypatch.delenv("INDOONE_TAVILY_API_KEY", raising=False)

    from app.ai.research import (
        DuckDuckGoResearchProvider,
        MultiSourceResearchProvider,
        SearXNGResearchProvider,
        build_research_provider,
    )

    provider = build_research_provider()

    assert isinstance(provider, MultiSourceResearchProvider)
    assert [type(item) for item in provider.providers] == [
        DuckDuckGoResearchProvider,
        SearXNGResearchProvider,
    ]


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


def test_build_research_query_variants_translates_roman_kannada() -> None:
    variants = build_research_query_variants("Karnataka da rajadhani yavudu?")
    assert variants[0] == "Karnataka da rajadhani yavudu?"
    assert variants[1] == "karnataka capital India"


def test_build_research_query_variants_extracts_latin_terms() -> None:
    variants = build_research_query_variants(
        "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ simple ಆಗಿ ಹೇಳು ಮತ್ತು sources ಕೊಡು."
    )

    assert variants[0].startswith("ಈಗಿನ AI technology")
    assert variants[1] == "AI technology"


def test_wikipedia_research_provider_adapts_knowledge_answer(monkeypatch) -> None:
    from app.ai.general_knowledge import WikipediaAnswer

    class FakeWikipedia:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def answer(self, query, language="English"):
            assert query == "Artificial intelligence"
            assert language == "English"
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


def test_multi_source_provider_filters_unrelated_broad_word_matches() -> None:
    class FakeProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            return [
                ResearchResult(
                    "AI technology advances in multimodal systems",
                    "https://relevant.example/ai",
                    "Recent AI technology research covers multimodal systems.",
                ),
                ResearchResult(
                    "What Is Perplexing AI?",
                    "https://unrelated.example/ai",
                    "An unrelated article that only happens to mention AI.",
                ),
                ResearchResult(
                    "Environmental costs threaten water and climate",
                    "https://another.example/environment",
                    "This result does not discuss AI technology.",
                ),
            ]

    provider = MultiSourceResearchProvider([FakeProvider()], max_query_variants=2)
    results = asyncio.run(
        provider.search(
            "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ simple ಆಗಿ explain ಮಾಡು ಮತ್ತು sources ಕೊಡು.",
            limit=8,
        )
    )

    assert [item.url for item in results] == ["https://relevant.example/ai"]


def test_score_research_result_ignores_generic_instruction_words() -> None:
    result = ResearchResult(
        "AI technology advances",
        "https://example.com/ai",
        "Research sources explain recent AI technology work.",
    )

    assert score_research_result(
        ["ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ simple ಆಗಿ sources ಕೊಡು"],
        result,
    ) >= 4.0


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
        "AI technology",
    ]
    assert news.queries == wikipedia.queries
    assert {item.url for item in results} == {
        "https://wikipedia.example/result",
        "https://news.example/result",
    }


def test_format_research_context_is_compact_and_preserves_source_identity() -> None:
    results = [
        ResearchResult(
            f"Source {index}",
            f"https://source{index}.example/result",
            "Evidence " + ("x" * 500),
        )
        for index in range(1, 9)
    ]

    formatted = format_research_context(results)

    assert formatted.startswith("<research>")
    assert formatted.endswith("</research>")
    for index in range(1, 9):
        assert f"source{index}.example" in formatted
        assert f"Source {index}" in formatted
    assert "x" * 500 not in formatted
    assert len(formatted) < 3_000


def test_score_research_result_prioritizes_title_matches() -> None:
    relevant = ResearchResult(
        "AI technology research",
        "https://relevant.example/item",
        "Unrelated background",
    )
    weak = ResearchResult(
        "History of computing",
        "https://weak.example/item",
        "Older computing history",
    )

    assert score_research_result(["AI technology research"], relevant) > score_research_result(
        ["AI technology research"], weak
    )


def test_research_relevance_supports_unicode_terms() -> None:
    query = ["ಭಾರತದ ಸ್ವಾತಂತ್ರ್ಯ ಯಾವಾಗ"]
    relevant = ResearchResult(
        "ಭಾರತದ ಸ್ವಾತಂತ್ರ್ಯ",
        "https://example.com/india",
        "ಭಾರತವು 1947ರಲ್ಲಿ ಸ್ವಾತಂತ್ರ್ಯ ಪಡೆದಿತು.",
    )
    unrelated = ResearchResult(
        "ಕ್ರೀಡೆ",
        "https://example.com/sports",
        "ಕ್ರೀಡೆ ಮತ್ತು ಪಂದ್ಯಗಳ ಕುರಿತು ಮಾಹಿತಿ.",
    )

    from app.ai.research import _is_relevant_research_result

    assert _is_relevant_research_result(query, relevant)
    assert not _is_relevant_research_result(query, unrelated)


def test_research_relevance_crosses_common_kannada_english_terms() -> None:
    query = ["ಭಾರತದ ರಾಷ್ಟ್ರಪತಿ"]
    result = ResearchResult(
        "President of India",
        "https://example.com/president",
        "The President of India is the head of state.",
    )
    assert _is_relevant_research_result(query, result)


def test_research_terms_keep_kannada_vowel_signs_attached() -> None:
    from app.ai.research import _research_terms

    assert "ಭಾರತದ" in _research_terms("ಭಾರತದ")
    assert "ರಾಷ್ಟ್ರಪತಿ" in _research_terms("ರಾಷ್ಟ್ರಪತಿ")
