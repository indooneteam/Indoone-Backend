import asyncio

import app.ai.service as service
from app.ai.intent import classify_intent
from app.ai.research import ResearchResult


def test_current_research_intent_requires_cross_check() -> None:
    latest = classify_intent("What is the latest Indoone news?")
    current = classify_intent("research current AI developments")
    stable = classify_intent("Explain how a transformer works")

    assert latest.needs_research is True
    assert latest.needs_cross_check is True
    assert current.needs_research is True
    assert current.needs_cross_check is True
    assert stable.needs_research is False
    assert stable.needs_cross_check is False


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
            return [
                ResearchResult("Indoone result 1", "https://example.com/indoone", "fresh source 1"),
                ResearchResult("Indoone result 2", "https://example.org/indoone", "fresh source 2"),
            ]

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", FakeResearch())
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())

    reply = asyncio.run(service.generate_reply("latest Indoone news"))

    assert reply.startswith("ok")
    assert "Sources:" in reply
    assert "https://example.com/indoone" in reply
    assert "https://example.org/indoone" in reply
    assert "<research>" in captured[0]
    assert "Indoone result 1" in captured[0]
    assert "Indoone result 2" in captured[0]
    assert "fresh source 1" in captured[0]
    assert "fresh source 2" in captured[0]


def test_single_source_is_not_treated_as_cross_checked(monkeypatch) -> None:
    captured: list[str] = []

    class FakeEngine:
        async def generate(self, message: str) -> str:
            captured.append(message)
            return "ok"

    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            return [ResearchResult("Only source", "https://example.com/indoone", "unverified")]

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", FakeResearch())
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())

    reply = asyncio.run(service.generate_reply("latest Indoone news"))

    assert reply == "ok"
    assert "Sources:" not in reply


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
