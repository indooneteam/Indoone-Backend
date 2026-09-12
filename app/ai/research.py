"""Optional live research layer for Indoone AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, urlparse

import httpx


MAX_TITLE_LENGTH = 500
MAX_SNIPPET_LENGTH = 2_000


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
        parsed = urlparse(base_url.strip())
        if not parsed.netloc or parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("timeout and max_response_bytes must be greater than zero")
        self.base_url = base_url.strip()
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


def build_research_provider() -> ResearchProvider | None:
    url = os.getenv("INDOONE_RESEARCH_URL", "").strip()
    if not url:
        return None
    try:
        timeout = float(os.getenv("INDOONE_RESEARCH_TIMEOUT", "10"))
    except ValueError as exc:
        raise ValueError("INDOONE_RESEARCH_TIMEOUT must be numeric") from exc
    return HttpResearchProvider(url, bearer_token=os.getenv("INDOONE_RESEARCH_TOKEN"), timeout=timeout)


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
