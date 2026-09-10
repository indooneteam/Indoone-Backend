"""Optional live research layer for Indoone AI.

Research is deliberately separate from the AI model. Indoone can use any
configured search service that returns JSON results; no hosted AI model is
required. When no research endpoint is configured, the layer stays disabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import httpx


@dataclass(frozen=True)
class ResearchResult:
    """A source returned by a research provider."""

    title: str
    url: str
    snippet: str = ""


class ResearchProvider:
    """Interface for live search providers."""

    async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
        raise NotImplementedError


class HttpResearchProvider(ResearchProvider):
    """Call a configured HTTP JSON search endpoint.

    The endpoint is expected to accept ``?q=<query>`` and return either a
    top-level list or ``{"results": [...]}``. Each result needs a title and
    URL; snippet is optional.
    """

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None = None,
        timeout: float = 10.0,
        max_response_bytes: int = 1_000_000,
    ) -> None:
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
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
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise RuntimeError("research response is too large")
            payload: Any = response.json()

        raw_results = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_results, list):
            raise RuntimeError("research response must contain a results list")

        results: list[ResearchResult] = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            source_url = str(item.get("url", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if title and source_url:
                results.append(ResearchResult(title=title, url=source_url, snippet=snippet))
        return results


def build_research_provider() -> ResearchProvider | None:
    """Build the configured live research provider, if enabled."""

    url = os.getenv("INDOONE_RESEARCH_URL", "").strip()
    if not url:
        return None
    return HttpResearchProvider(
        url,
        bearer_token=os.getenv("INDOONE_RESEARCH_TOKEN"),
        timeout=float(os.getenv("INDOONE_RESEARCH_TIMEOUT", "10")),
    )


def format_results(results: list[ResearchResult]) -> str:
    """Format sources for model context without losing provenance."""

    if not results:
        return ""
    lines = ["<research>"]
    for result in results:
        lines.append(f"title: {result.title}")
        lines.append(f"url: {result.url}")
        if result.snippet:
            lines.append(f"snippet: {result.snippet}")
    lines.append("</research>")
    return "\n".join(lines)
