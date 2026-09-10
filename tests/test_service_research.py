import asyncio

import app.ai.service as service
from app.ai.research import ResearchResult


def test_should_research_detects_freshness_requests() -> None:
    assert service.should_research("What is the latest Indoone news?")
    assert service.should_research("research current AI developments")
    assert not service.should_research("Explain how a transformer works")


def test_service_includes_research_results(monkeypatch) -> None:
    captured: list[str] = []

    class FakeEngine:
        async def generate(self, message: str) -> str:
            captured.append(message)
            return "ok"

    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest Indoone news"
            assert limit == 5
            return [ResearchResult("Indoone result", "https://example.com/indoone", "fresh source")]

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", FakeResearch())
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())

    reply = asyncio.run(service.generate_reply("latest Indoone news"))

    assert reply == "ok"
    assert "<research>" in captured[0]
    assert "Indoone result" in captured[0]
    assert "https://example.com/indoone" in captured[0]
    assert "fresh source" in captured[0]


def test_service_survives_research_failure(monkeypatch) -> None:
    captured: list[str] = []

    class FakeEngine:
        async def generate(self, message: str) -> str:
            captured.append(message)
            return "ok"

    class BrokenResearch:
        async def search(self, query: str, limit: int = 5):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", BrokenResearch())
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())

    reply = asyncio.run(service.generate_reply("latest Indoone news"))

    assert reply == "ok"
    assert "<research>" not in captured[0]
