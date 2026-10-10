import pytest

from app.ai import service
from app.web_research.research import ResearchResult


@pytest.mark.asyncio
async def test_service_collects_research_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            calls.append(("research", query))
            return [
                ResearchResult("Source one", "https://example.com/one", "Evidence one."),
                ResearchResult("Source two", "https://example.org/two", "Evidence two."),
            ]

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            calls.append(("generation", user_prompt))
            return "Generated from live evidence."

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", FakeResearch())
    monkeypatch.setattr(service, "_knowledge_base", None)

    reply = await service.generate_reply("latest information about a topic")

    assert reply.startswith("Generated from live evidence.")
    assert calls[0] == ("research", "latest information about a topic")
    assert calls[1][0] == "generation"
    assert "Evidence one." in calls[1][1]


@pytest.mark.asyncio
async def test_service_does_not_guess_when_current_research_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenResearch:
        async def search(self, query: str, limit: int = 5):
            raise RuntimeError("provider unavailable")

    class FakeProvider:
        async def generate(self, **kwargs):
            raise AssertionError("generation must not fabricate a current answer")

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", BrokenResearch())
    monkeypatch.setattr(service, "_knowledge_base", None)

    with pytest.raises(RuntimeError, match="live research is unavailable"):
        await service.generate_reply("latest information about a topic")
