from __future__ import annotations

import asyncio

import httpx

from app.web_research import tools
from app.web_research.research import ResearchResult


def _call(query: str, content=None, call_id: str | None = None) -> tools.SearchWebToolCall:
    return tools.SearchWebToolCall(
        query=query,
        model_content=content or {"parts": [{"functionCall": {"name": "search_web", "args": {"query": query}}}]},
        call_id=call_id,
    )


def test_tool_is_available_for_both_gemma4_models() -> None:
    for model in ("gemma-4-26b-a4b-it", "gemma-4-31b-it"):
        declaration = tools.build_search_web_tools(model)[0]["functionDeclarations"][0]
        assert declaration["name"] == "search_web"
        assert declaration["parameters"]["required"] == ["query"]


def test_tool_is_not_enabled_for_other_models() -> None:
    assert tools.build_search_web_tools("custom-model") == []


def test_extract_search_call_preserves_content_and_id() -> None:
    content = {"parts": [{"functionCall": {"name": "search_web", "id": "call-1", "args": {"query": "  latest Indoone news  "}}}]}
    call = tools.extract_search_web_call({"candidates": [{"content": content}]})
    assert call is not None
    assert call.query == "latest Indoone news"
    assert call.call_id == "call-1"
    assert call.model_content is content


def test_extract_search_call_ignores_normal_text_and_unknown_functions() -> None:
    assert tools.extract_search_web_call({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}) is None
    assert tools.extract_search_web_call({"candidates": [{"content": {"parts": [{"functionCall": {"name": "send_email", "args": {}}}]}}]}) is None


def test_execute_uses_free_public_provider(monkeypatch) -> None:
    class FakeProvider:
        async def search(self, query: str, limit: int = 5):
            assert query == "latest Indoone news"
            assert limit == tools.MAX_SEARCH_RESULTS
            return [ResearchResult("Official update", "https://example.com/news", "Fresh facts")]

    monkeypatch.setattr(tools, "build_free_research_provider", lambda: FakeProvider())
    result = asyncio.run(tools.execute_search_web_call(_call("latest Indoone news")))
    assert result.status == "ok"
    assert result.results == (ResearchResult("Official update", "https://example.com/news", "Fresh facts"),)


def test_execute_rejects_empty_oversized_or_private_queries(monkeypatch) -> None:
    def forbidden_provider():
        raise AssertionError("invalid queries must never access search")

    monkeypatch.setattr(tools, "build_free_research_provider", forbidden_provider)
    queries = ("  ", "x" * (tools.MAX_SEARCH_QUERY_LENGTH + 1), "Search person@example.com", "API_KEY=secret")
    for query in queries:
        assert asyncio.run(tools.execute_search_web_call(_call(query))).status == "invalid_query"


def test_provider_failure_is_reported_without_fabricating_sources(monkeypatch) -> None:
    class BrokenProvider:
        async def search(self, query: str, limit: int = 5):
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(tools, "build_free_research_provider", lambda: BrokenProvider())
    result = asyncio.run(tools.execute_search_web_call(_call("latest Indoone news")))
    assert result.status == "unavailable"
    assert result.results == ()


def test_function_response_contains_call_id_and_sources() -> None:
    call = _call("latest news", call_id="call-42")
    result = tools.SearchWebToolResult(
        status="ok",
        results=(ResearchResult("News source", "https://example.com/news", "Snippet"),),
    )
    output = tools.build_search_web_function_response(call, result)
    assert output["role"] == "user"
    fn = output["parts"][0]["functionResponse"]
    assert fn["name"] == "search_web"
    assert fn["id"] == "call-42"
    assert fn["response"]["results"] == [
        {"title": "News source", "url": "https://example.com/news", "snippet": "Snippet"}
    ]


def test_unavailable_function_response_tells_model_not_to_claim_verification() -> None:
    output = tools.build_search_web_function_response(
        _call("latest news"), tools.SearchWebToolResult(status="unavailable")
    )
    assert "Do not claim current facts were verified" in output["parts"][0]["functionResponse"]["response"]["message"]
