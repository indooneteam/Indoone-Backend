"""Google Search grounding helpers for supported Gemma 4 models."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.web_research.research import ResearchResult


def build_google_search_tools(model: str) -> list[dict[str, Any]]:
    """Expose Google Search grounding to supported Gemma 4 models.

    The model can decide whether a question needs a search. Other model families
    are left unchanged until their supported tool matrix is configured explicitly.
    """
    normalized = model.strip().casefold()
    if normalized.startswith("gemma-4-"):
        return [{"googleSearch": {}}]
    return []


def extract_grounding_sources(response_data: dict[str, Any]) -> list[ResearchResult]:
    """Extract and validate source links returned by Gemini grounding metadata."""
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return []
    metadata = candidate.get("groundingMetadata")
    if not isinstance(metadata, dict):
        return []
    chunks = metadata.get("groundingChunks")
    if not isinstance(chunks, list):
        return []

    results: list[ResearchResult] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        title = " ".join(str(web.get("title", "")).split()).strip()[:500]
        url = str(web.get("uri", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not title:
            continue
        key = url.rstrip("/").casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(ResearchResult(title=title, url=url, snippet=""))
    return results


def format_sources_footer(answer: str, sources: list[ResearchResult]) -> str:
    """Append a compact source list so every channel receives grounding links."""
    clean_answer = answer.rstrip()
    if not sources:
        return clean_answer
    lines = [clean_answer, "", "Sources:"]
    for source in sources:
        title = " ".join(source.title.split()).strip()[:500]
        url = source.url.strip()
        parsed = urlparse(url)
        if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines) if len(lines) > 3 else clean_answer
