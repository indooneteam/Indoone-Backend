import asyncio

import app.ai.service as service
from app.ai.research import ResearchResult


def test_service_appends_research_sources(monkeypatch) -> None:
    captured: list[str] = []

    class FakeEngine:
        async def generate(self, message: str) -> str:
            captured.append(message)
            return "grounded answer"

    class FakeProvider:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest Indoone news"
            assert limit == 5
            return [
                ResearchResult(
                    "Indoone source 1",
                    "https://example.com/indoone",
                    "source snippet 1",
                ),
                ResearchResult(
                    "Indoone source 2",
                    "https://example.org/indoone",
                    "source snippet 2",
                ),
            ]

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())
    monkeypatch.setattr(service, "_research_provider", FakeProvider())

    reply = asyncio.run(service.generate_reply("latest Indoone news"))

    assert reply == (
        "grounded answer\n\nSources:\n"
        "1. Indoone source 1 — https://example.com/indoone\n"
        "2. Indoone source 2 — https://example.org/indoone"
    )
    assert captured[0].startswith("<instruction>")
    assert "Use the supplied knowledge and research evidence when relevant." in captured[0]
    assert "<research>" in captured[0] or "research" in captured[0]
    assert "https://example.com/indoone" in captured[0]
    assert "https://example.org/indoone" in captured[0]
    assert captured[0].endswith("<response>")
