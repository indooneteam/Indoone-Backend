"""Optional live research layer for Indoone AI."""

from __future__ import annotations


import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, urlparse

import httpx

MAX_TITLE_LENGTH = 500
MAX_SNIPPET_LENGTH = 2_000
MAX_RESEARCH_QUERIES = 6
MAX_RESEARCH_TIMEOUT_SECONDS = 60.0
DEFAULT_MULTI_SOURCE_LIMIT = 8
MAX_QUERY_VARIANTS = 2

@dataclass(frozen=True)
class ResearchResult:
    """A source returned by a research provider."""
    title: str
    url: str
    snippet: str = ""

class ResearchProvider:
    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        raise NotImplementedError

def _safe_source_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value.strip()

def _sanitize_result(item: Any) -> ResearchResult | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip()[:MAX_TITLE_LENGTH]
    source_url = _safe_source_url(str(item.get("url", "")))
    snippet = str(item.get("snippet", "")).strip()[:MAX_SNIPPET_LENGTH]
    if not title or not source_url:
        return None
    return ResearchResult(title=title, url=source_url, snippet=snippet)

class HttpResearchProvider(ResearchProvider):
    def __init__(self, base_url: str, bearer_token: str | None = None, timeout: float = 10.0, max_response_bytes: int = 1_000_000) -> None:
        base_url = base_url.strip()
        parsed = urlparse(base_url)
        if not base_url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("timeout and max_response_bytes must be greater than zero")
        self.base_url = base_url
        self.bearer_token = bearer_token.strip() if bearer_token else None
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        separator = "&" if "?" in self.base_url else "?"
        url = f"{self.base_url}{separator}q={quote_plus(query)}&limit={limit}"
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            payload: Any = response.json()
        raw_results = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_results, list):
            raise RuntimeError("research response must contain a results list")
        return [result for item in raw_results[:limit] if (result := _sanitize_result(item)) is not None]


def build_research_query_variants(query: str, max_variants: int = MAX_QUERY_VARIANTS) -> list[str]:
    """Build a small set of resilient queries for mixed-language user prompts."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query cannot be empty")
    if max_variants < 1 or max_variants > MAX_QUERY_VARIANTS:
        raise ValueError(f"max_variants must be between 1 and {MAX_QUERY_VARIANTS}")

    variants = [normalized]
    latin_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*", normalized)
    english_variant = " ".join(latin_terms).strip()
    if english_variant and english_variant.casefold() != normalized.casefold():
        variants.append(english_variant)
    return variants[:max_variants]


class WikipediaResearchProvider(ResearchProvider):
    """Adapt the existing Wikipedia knowledge provider into research evidence."""

    def __init__(self, timeout: float = 6.0) -> None:
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        self.timeout = timeout

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        from app.ai.general_knowledge import WikipediaKnowledgeProvider

        provider = WikipediaKnowledgeProvider(timeout=self.timeout)
        answer = await provider.answer(query)
        if answer is None:
            return []
        return [ResearchResult(answer.title, answer.url, answer.extract)]


class WikidataResearchProvider(ResearchProvider):
    """Search Wikidata entities and return their labels/descriptions as evidence."""

    _BASE_URL = "https://www.wikidata.org/w/api.php"

    def __init__(self, timeout: float = 8.0, max_response_bytes: int = 750_000) -> None:
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        headers = {
            "Accept": "application/json",
            "User-Agent": "Indoone-Research/1.0",
        }
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "type": "item",
            "limit": str(limit),
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.get(self._BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            payload = response.json()

        items = payload.get("search", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []

        results: list[ResearchResult] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("id", "")).strip()
            label = str(item.get("label", "")).strip()
            description = str(item.get("description", "")).strip()
            if not entity_id or not label:
                continue
            url = _safe_source_url(f"https://www.wikidata.org/wiki/{entity_id}")
            if not url:
                continue
            results.append(
                ResearchResult(label[:MAX_TITLE_LENGTH], url, description[:MAX_SNIPPET_LENGTH])
            )
        return results


class OpenAlexResearchProvider(ResearchProvider):
    """Search OpenAlex works for research literature."""

    _BASE_URL = "https://api.openalex.org/works"

    def __init__(self, timeout: float = 8.0, max_response_bytes: int = 1_500_000) -> None:
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def _abstract_from_inverted_index(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        tokens: list[tuple[int, str]] = []
        for word, positions in value.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int) and position >= 0:
                    tokens.append((position, word))
        tokens.sort(key=lambda item: item[0])
        return " ".join(word for _, word in tokens)

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        params = {
            "search": query,
            "per-page": str(limit),
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "Indoone-Research/1.0",
        }
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.get(self._BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            payload = response.json()

        items = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []

        results: list[ResearchResult] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("display_name", "")).strip() or str(item.get("title", "")).strip()
            primary_location = item.get("primary_location")
            primary_url = (
                str(primary_location.get("landing_page_url", "")).strip()
                if isinstance(primary_location, dict)
                else ""
            )
            doi = str(item.get("doi", "")).strip()
            work_id = str(item.get("id", "")).strip().rsplit("/", 1)[-1]
            source_url = primary_url or doi or (f"https://openalex.org/{work_id}" if work_id else "")
            source_url = _safe_source_url(source_url)
            if not title or not source_url:
                continue
            abstract = self._abstract_from_inverted_index(item.get("abstract_inverted_index"))
            snippet = abstract or str(item.get("publication_year", "")).strip()
            results.append(
                ResearchResult(title[:MAX_TITLE_LENGTH], source_url, snippet[:MAX_SNIPPET_LENGTH])
            )
        return results


class CrossrefResearchProvider(ResearchProvider):
    """Search Crossref for published scholarly works."""

    _BASE_URL = "https://api.crossref.org/works"

    def __init__(self, timeout: float = 8.0, max_response_bytes: int = 1_500_000) -> None:
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        params = {
            "query": query,
            "rows": str(limit),
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "Indoone-Research/1.0",
        }
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.get(self._BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            payload = response.json()

        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        items = message.get("items", []) if isinstance(message, dict) else []
        if not isinstance(items, list):
            return []

        results: list[ResearchResult] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            raw_title = item.get("title", [])
            title = str(raw_title[0]).strip() if isinstance(raw_title, list) and raw_title else ""
            source_url = _safe_source_url(str(item.get("URL", "")).strip())
            if not title or not source_url:
                continue
            abstract = str(item.get("abstract", "")).strip()
            abstract = re.sub(r"<[^>]+>", " ", abstract)
            abstract = " ".join(abstract.split())
            results.append(
                ResearchResult(title[:MAX_TITLE_LENGTH], source_url, abstract[:MAX_SNIPPET_LENGTH])
            )
        return results


class MultiSourceResearchProvider(ResearchProvider):
    """Fan out across independent research sources and return one merged set."""

    def __init__(
        self,
        providers: list[ResearchProvider] | None = None,
        *,
        max_query_variants: int = MAX_QUERY_VARIANTS,
    ) -> None:
        self.providers = providers or [
            WikipediaResearchProvider(),
            WikidataResearchProvider(),
            GoogleNewsRssResearchProvider(),
            OpenAlexResearchProvider(),
            CrossrefResearchProvider(),
        ]
        if not self.providers:
            raise ValueError("providers cannot be empty")
        if max_query_variants < 1 or max_query_variants > MAX_QUERY_VARIANTS:
            raise ValueError(f"max_query_variants must be between 1 and {MAX_QUERY_VARIANTS}")
        self.max_query_variants = max_query_variants

    async def search(self, query: str, limit: int = DEFAULT_MULTI_SOURCE_LIMIT) -> list[ResearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        variants = build_research_query_variants(query, max_variants=self.max_query_variants)
        per_provider_limit = min(3, limit)
        tasks = [
            provider.search(variant, limit=per_provider_limit)
            for provider in self.providers
            for variant in variants
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        # Keep a small result pool for each provider so one source cannot fill the
        # whole answer before the other independent sources contribute evidence.
        provider_results: list[list[ResearchResult]] = [[] for _ in self.providers]
        outcome_index = 0
        for provider_index in range(len(self.providers)):
            per_provider_seen: set[str] = set()
            for _ in variants:
                outcome = outcomes[outcome_index]
                outcome_index += 1
                if isinstance(outcome, Exception):
                    continue
                for item in outcome:
                    key = item.url.strip().rstrip("/").casefold()
                    if key and key not in per_provider_seen:
                        per_provider_seen.add(key)
                        provider_results[provider_index].append(item)

        merged: list[ResearchResult] = []
        seen: set[str] = set()
        while len(merged) < limit and any(provider_results):
            progressed = False
            for results in provider_results:
                if not results:
                    continue
                item = results.pop(0)
                key = item.url.strip().rstrip("/").casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                progressed = True
                if len(merged) >= limit:
                    break
            if not progressed:
                break
        return merged


class GoogleNewsRssResearchProvider(ResearchProvider):
    """Keyless live research provider backed by the public Google News RSS feed."""

    _BASE_URL = "https://news.google.com/rss/search"

    def __init__(self, timeout: float = 10.0, max_response_bytes: int = 1_000_000) -> None:
        if timeout <= 0 or timeout > MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout must not exceed 60 seconds and must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        from html import unescape
        from xml.etree import ElementTree

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        url = (
            f"{self._BASE_URL}?q={quote_plus(query)}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        headers = {
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            "User-Agent": "Indoone-Research/1.0",
        }
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            content = response.content

        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            raise RuntimeError("research response is not valid RSS/XML") from exc

        results: list[ResearchResult] = []
        seen_urls: set[str] = set()
        for item in root.findall("./channel/item"):
            title = unescape((item.findtext("title") or "").strip())
            source_url = _safe_source_url((item.findtext("link") or "").strip())
            description = unescape((item.findtext("description") or "").strip())
            description = re.sub(r"<[^>]+>", " ", description)
            description = " ".join(description.split())[:MAX_SNIPPET_LENGTH]
            if not title or not source_url or source_url in seen_urls:
                continue
            seen_urls.add(source_url)
            results.append(
                ResearchResult(
                    title=title[:MAX_TITLE_LENGTH],
                    url=source_url,
                    snippet=description,
                )
            )
            if len(results) >= limit:
                break
        return results

def build_research_provider() -> ResearchProvider | None:
    try:
        timeout = float(os.getenv("INDOONE_RESEARCH_TIMEOUT", "8"))
    except ValueError as exc:
        raise ValueError("INDOONE_RESEARCH_TIMEOUT must be numeric") from exc

    providers: list[ResearchProvider] = [
        WikipediaResearchProvider(timeout=min(timeout, 6.0)),
        WikidataResearchProvider(timeout=timeout),
        GoogleNewsRssResearchProvider(timeout=timeout),
        OpenAlexResearchProvider(timeout=timeout),
        CrossrefResearchProvider(timeout=timeout),
    ]

    url = os.getenv("INDOONE_RESEARCH_URL", "").strip()
    if url:
        providers.insert(
            0,
            HttpResearchProvider(
                url,
                bearer_token=os.getenv("INDOONE_RESEARCH_TOKEN"),
                timeout=timeout,
            ),
        )

    return MultiSourceResearchProvider(providers)
def build_deep_research_queries(query: str, count: int = 3) -> list[str]:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query cannot be empty")
    if count < 1 or count > MAX_RESEARCH_QUERIES:
        raise ValueError(f"count must be between 1 and {MAX_RESEARCH_QUERIES}")
    candidates = [
        normalized,
        f"{normalized} official sources",
        f"{normalized} recent developments",
        f"{normalized} data statistics evidence",
        f"{normalized} risks limitations criticism",
        f"{normalized} alternatives comparison",
    ]
    return candidates[:count]

def merge_research_results(results_by_query: dict[str, list[ResearchResult]], limit: int = 50) -> list[dict[str, Any]]:
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")
    merged: dict[str, dict[str, Any]] = {}
    for query, results in results_by_query.items():
        for item in results:
            existing = merged.get(item.url)
            if existing is None:
                merged[item.url] = {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "queries": [query],
                }
            else:
                if query not in existing["queries"]:
                    existing["queries"].append(query)
                if len(item.snippet) > len(existing["snippet"]):
                    existing["snippet"] = item.snippet
    return sorted(merged.values(), key=lambda item: (-len(item["queries"]), item["url"]))[:limit]

def format_research_context(
    results: list[ResearchResult],
    *,
    max_results: int = DEFAULT_MULTI_SOURCE_LIMIT,
    max_title_chars: int = 80,
    max_snippet_chars: int = 180,
) -> str:
    """Create a compact evidence block that fits the local model context window.

    Full URLs remain available for the final user-facing Sources section. The
    model only needs source identity plus a short evidence snippet for synthesis.
    """
    if max_results < 1 or max_results > 20:
        raise ValueError("max_results must be between 1 and 20")
    if max_title_chars < 1 or max_snippet_chars < 1:
        raise ValueError("context limits must be greater than zero")

    lines = ["<research>"]
    for result in results[:max_results]:
        title = " ".join(result.title.split())[:max_title_chars]
        domain = urlparse(result.url).netloc.casefold()
        snippet = " ".join(result.snippet.split())[:max_snippet_chars]
        if not title or not domain:
            continue
        lines.append(f"source: {domain}")
        lines.append(f"title: {title}")
        if snippet:
            lines.append(f"snippet: {snippet}")
    lines.append("</research>")
    return "\n".join(lines) if len(lines) > 1 else ""

def format_results(results: list[ResearchResult | dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["<research>"]
    for result in results:
        queries: Any = None
        if isinstance(result, ResearchResult):
            title, url, snippet = result.title, result.url, result.snippet
        else:
            title = str(result.get("title", "")).strip()
            url = str(result.get("url", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            queries = result.get("queries")
        lines.append(f"title: {title}")
        lines.append(f"url: {url}")
        if snippet:
            lines.append(f"snippet: {snippet}")
        if isinstance(queries, list) and queries:
            lines.append("found_by: " + " | ".join(str(item) for item in queries))
    lines.append("</research>")
    return "\n".join(lines)
