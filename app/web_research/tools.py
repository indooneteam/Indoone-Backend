"""Gemma 4 optional search function, implemented with keyless public providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import httpx

from app.web_research.research import ResearchResult, build_free_research_provider

MAX_SEARCH_QUERY_LENGTH = 500
MAX_SEARCH_RESULTS = 8
_SEARCH_TOOL_NAME = "search_web"
_PRIVATE_QUERY_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:password|api[_ -]?key|access[_ -]?token|secret)\s*[:=]", re.IGNORECASE),
)


@dataclass(frozen=True)
class SearchWebToolCall:
    query: str
    model_content: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class SearchWebToolResult:
    status: str
    results: tuple[ResearchResult, ...] = ()


def build_search_web_tools(model: str) -> list[dict[str, Any]]:
    """Expose a search tool to the Gemma 4 models configured for Indoone."""
    if not model.strip().casefold().startswith("gemma-4-"):
        return []
    return [{
        "functionDeclarations": [{
            "name": _SEARCH_TOOL_NAME,
            "description": (
                "Search public web sources only when the user needs current/recent facts, latest news, "
                "prices, availability, or explicitly requests online research. Do not call for stable "
                "evergreen questions. Never search private messages, email addresses, passwords, API keys, "
                "access tokens, or secrets."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "A concise public web query for the user's question.",
                    }
                },
                "required": ["query"],
            },
        }]
    }]


def extract_search_web_call(response_data: dict[str, Any]) -> SearchWebToolCall | None:
    """Parse a model-selected search_web call and preserve its full content."""
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    content = candidate.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        if not isinstance(part, dict):
            continue
        fn = part.get("functionCall")
        if not isinstance(fn, dict) or fn.get("name") != _SEARCH_TOOL_NAME:
            continue
        args = fn.get("args")
        raw_query = args.get("query") if isinstance(args, dict) else None
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        raw_id = fn.get("id")
        call_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None
        return SearchWebToolCall(query=query, model_content=content, call_id=call_id)
    return None


async def execute_search_web_call(call: SearchWebToolCall) -> SearchWebToolResult:
    """Run one bounded query with keyless public providers only."""
    query = " ".join(call.query.split())
    if (
        not query
        or len(query) > MAX_SEARCH_QUERY_LENGTH
        or any(pattern.search(query) for pattern in _PRIVATE_QUERY_PATTERNS)
    ):
        return SearchWebToolResult(status="invalid_query")
    try:
        provider = build_free_research_provider()
        results = await provider.search(query, limit=MAX_SEARCH_RESULTS)
    except (RuntimeError, ValueError, httpx.HTTPError):
        return SearchWebToolResult(status="unavailable")
    return SearchWebToolResult(
        status="ok" if results else "no_results",
        results=tuple(results[:MAX_SEARCH_RESULTS]),
    )


def build_search_web_function_response(
    call: SearchWebToolCall,
    result: SearchWebToolResult,
) -> dict[str, Any]:
    """Format public search evidence as a Gemini function-response turn."""
    response: dict[str, Any] = {
        "status": result.status,
        "results": [
            {"title": item.title, "url": item.url, "snippet": item.snippet}
            for item in result.results
        ],
    }
    if result.status == "unavailable":
        response["message"] = "Public web search was unavailable. Do not claim current facts were verified."
    elif result.status == "no_results":
        response["message"] = "No relevant public sources were found. Do not invent sources."
    elif result.status == "invalid_query":
        response["message"] = "The query was invalid or contained private data. Do not search it."
    function_response: dict[str, Any] = {"name": _SEARCH_TOOL_NAME, "response": response}
    if call.call_id:
        function_response["id"] = call.call_id
    # GenerateContent expects function responses inside a user-role Content turn.
    return {"role": "user", "parts": [{"functionResponse": function_response}]}
