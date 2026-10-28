"""Gemma 4 optional search function, implemented with keyless public providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import httpx

from app.web_research.research import ResearchResult, build_free_research_provider
from app.ai.gateway.ai_rules import SEARCH_INVALID_QUERY, SEARCH_NO_RESULTS, SEARCH_UNAVAILABLE

MAX_SEARCH_QUERY_LENGTH = 500
MAX_SEARCH_RESULTS = 8
_SEARCH_TOOL_NAME = "search_web"
_PRIVATE_QUERY_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:password|api[_ -]?key|access[_ -]?token|secret)\s*[:=]", re.IGNORECASE),
)


# "web research maadu", "search the web for ...", "internet alli huduku", "ಜಾಲದಲ್ಲಿ ಹುಡುಕು".
# A message that asks for research or names the web is asking for a search, and the
# declaration must not depend on the model choosing to discover the tool first - it did not,
# and answered from training memory instead.
#
# Kannada is matched without word boundaries: it agglutinates, so "ಜಾಲ" arrives as
# "ಜಾಲದಲ್ಲಿ" and a \b on either side of the stem would never fire.
_RESEARCH_ASK = re.compile(r"\b(research|researching)\b|huduku|huduk|ಸಂಶೋಧ|ಹುಡುಕ", re.IGNORECASE)
_WEB_PLACE = re.compile(r"\b(web|internet|online|google)\b|ವೆಬ್|ಇಂಟರ್ನೆಟ್|ಜಾಲ|ನೆಟ್", re.IGNORECASE)
_NEWS_ASK = re.compile(r"\bnews\b|ಸುದ್ದಿ", re.IGNORECASE)
_SEARCH_VERB = re.compile(r"\b(search|find|look\s*up|google)\b|ಹುಡುಕ|huduku|huduk", re.IGNORECASE)


def wants_web_research(message: str) -> bool:
    """True when the message asks for a web search in so many words.

    Deliberately narrow: research, news, or a place (web, internet, online, Google) together
    with a search verb. "search my emails" and "search the repository" name a different
    place, and a bare "search" does not mean the web.
    """
    text = (message or "").strip()

    if not text:
        return False

    if _RESEARCH_ASK.search(text) or _NEWS_ASK.search(text):
        return True

    return bool(_WEB_PLACE.search(text)) and bool(_SEARCH_VERB.search(text))


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
                "Search public web sources when the user needs current or recent facts, latest news, "
                "prices, availability, or explicitly requests online research. Festival and holiday dates, "
                "religious dates, auspicious times (muhurta), tithi and panchang are current information "
                "because they change every year - search for them rather than refusing. Do not call for "
                "stable evergreen questions. Never search private messages, email addresses, passwords, "
                "API keys, access tokens, or secrets."
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
        response["message"] = SEARCH_UNAVAILABLE
    elif result.status == "no_results":
        response["message"] = SEARCH_NO_RESULTS
    elif result.status == "invalid_query":
        response["message"] = SEARCH_INVALID_QUERY
    function_response: dict[str, Any] = {"name": _SEARCH_TOOL_NAME, "response": response}
    if call.call_id:
        function_response["id"] = call.call_id
    # GenerateContent expects function responses inside a user-role Content turn.
    return {"role": "user", "parts": [{"functionResponse": function_response}]}
