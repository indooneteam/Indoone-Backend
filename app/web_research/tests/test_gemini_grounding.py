"""Gemma 4 decides when to use the free public web-search tool."""

from __future__ import annotations

import asyncio

from app.ai import gemini_service
from app.web_research.research import ResearchResult
from app.web_research.tools import SearchWebToolResult


def _response(parts: list[dict[str, object]]) -> dict[str, object]:
    return {"candidates": [{"content": {"parts": parts}}]}


def test_model_selected_search_is_executed_and_sources_are_returned(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    first = _response([{
        "functionCall": {
            "name": "search_web",
            "id": "search-1",
            "args": {"query": "latest Indoone update"},
        }
    }])
    final = _response([{"text": "The official update is available now."}])

    class FakeResponse:
        status_code = 200
        text = ""
        def __init__(self, payload):
            self.payload = payload
        def json(self):
            return self.payload

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None
        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]):
            calls.append(json)
            if len(calls) == 1:
                assert json["tools"][0]["functionDeclarations"][0]["name"] == "search_web"
                assert "googleSearch" not in json["tools"][0]
                return FakeResponse(first)
            assert len(calls) == 2
            assert "tools" not in json
            assert json["contents"][-2]["role"] == "model"
            assert json["contents"][-1]["role"] == "user"
            function_response = json["contents"][-1]["parts"][0]["functionResponse"]
            assert function_response["response"]["results"][0]["url"] == "https://example.com/update"
            return FakeResponse(final)

    async def fake_execute(call):
        assert call.query == "latest Indoone update"
        return SearchWebToolResult(
            status="ok",
            results=(ResearchResult("Official update", "https://example.com/update", "The official update is available now."),),
        )

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service, "execute_search_web_call", fake_execute)
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)
    reply = asyncio.run(gemini_service.generate_gemini_reply("What is the latest update?"))

    assert reply == "\n".join([
        "The official update is available now.",
        "",
        "Sources:",
        "1. Official update — https://example.com/update",
    ])


def test_stable_question_does_not_trigger_search_when_model_skips_tool(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return _response([{"text": "Photosynthesis converts light into chemical energy."}])

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None
        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]):
            calls.append(json)
            return FakeResponse()

    def forbidden_execute(call):
        raise AssertionError("The model did not call search_web.")

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service, "execute_search_web_call", forbidden_execute)
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)
    reply = asyncio.run(gemini_service.generate_gemini_reply("Explain photosynthesis."))

    assert reply == "Photosynthesis converts light into chemical energy."
    assert len(calls) == 1
