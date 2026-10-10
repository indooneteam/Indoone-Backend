"""HTTP endpoints for live and deep web research."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.web_research.research import build_free_research_provider

router = APIRouter(tags=["web-research"])


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=5, ge=1, le=20)


class DeepResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1_000)
    queries: int = Field(default=3, ge=1, le=5)
    per_query_limit: int = Field(default=5, ge=1, le=10)


@router.post("/research")
async def research(request: ResearchRequest) -> dict[str, object]:
    provider = build_free_research_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="live research provider is not configured")
    try:
        results = await provider.search(request.query, request.limit)
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=f"research provider failed: {exc}") from exc
    return {
        "query": request.query,
        "results": [
            {"title": item.title, "url": item.url, "snippet": item.snippet}
            for item in results
        ],
    }


@router.post("/deep-research")
async def deep_research(request: DeepResearchRequest) -> dict[str, object]:
    provider = build_free_research_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="live research provider is not configured")
    queries = [
        request.query,
        f"{request.query} official sources",
        f"{request.query} recent developments",
    ][: request.queries]
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            batch = await provider.search(query, request.per_query_limit)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=f"research provider failed: {exc}") from exc
        for item in batch:
            if item.url in seen:
                continue
            seen.add(item.url)
            results.append(
                {"query": query, "title": item.title, "url": item.url, "snippet": item.snippet}
            )
    return {"query": request.query, "queries": queries, "sources": results}
