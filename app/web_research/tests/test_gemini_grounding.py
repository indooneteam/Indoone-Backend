"""Tests that Gemini uses keyless public web research selectively."""

from __future__ import annotations

import asyncio

from app.ai import gemini_service
from app.web_research.google_search import should_use_live_research
from app.web_research.research import ResearchResult


def test_live_research_intent_is_selective_and_multilingual() -> None:
    assert should_use_live_research("What is the latest Indoone update?")
    assert should_use_live_research("ಇವತ್ತಿನ ಸುದ್ದಿ ಏನು?")
    assert should_use_live_research("ivattina suddi heli")
    assert not should_use_live_research("Explain photosynthesis simply")


def test_gemini_reply_uses_free_public_sources_for_live_queries(monkeypatch) -> None:
    class FakeProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "What is the latest update?"
            assert limit == 8
            return [ResearchResult("Official update", "https://example.com/update", "Verified current evidence.")]

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {"candidates": [{"content": {"parts": [{"text": "A current answer."}]}}]}

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            assert "googleSearch" not in json.get("tools", [{}])[0] if json.get("tools") else True
            user_text = json["contents"][-1]["parts"][0]["text"]
            assert "LIVE WEB RESEARCH EVIDENCE:" in user_text
            assert "Verified current evidence." in user_text
            return FakeResponse()

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service, "build_free_research_provider", lambda: FakeProvider())
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)
    reply = asyncio.run(gemini_service.generate_gemini_reply("What is the latest update?"))
    assert reply == "A current answer.\\n\\nSources:\\n- [Official update](https://example.com/update)"


def test_gemini_reply_skips_search_for_stable_questions(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {"candidates": [{"content": {"parts": [{"text": "A stable answer."}]}}]}

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            assert "LIVE WEB RESEARCH EVIDENCE:" not in json["contents"][-1]["parts"][0]["text"]
            return FakeResponse()

    def forbidden_provider():
        raise AssertionError("stable questions must not trigger web research")

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service, "build_free_research_provider", forbidden_provider)
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)
    assert asyncio.run(gemini_service.generate_gemini_reply("Explain photosynthesis")) == "A stable answer."
